"""
SQLite-based critique pattern store for self-improvement.

Stores successful critique patterns so future debates can learn from past successes.
"""

import sqlite3
import json
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Generator

from aragora.core import Critique, DebateResult


# Database connection timeout in seconds
DB_TIMEOUT = 30.0


@dataclass
class Pattern:
    """A reusable critique pattern."""

    id: str
    issue_type: str  # categorized issue type
    issue_text: str
    suggestion_text: str
    success_count: int
    failure_count: int
    avg_severity: float
    example_task: str
    created_at: str
    updated_at: str

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5


@dataclass
class AgentReputation:
    """Per-agent reputation tracking for weighted voting."""

    agent_name: str
    proposals_made: int = 0
    proposals_accepted: int = 0
    critiques_given: int = 0
    critiques_valuable: int = 0
    updated_at: str = ""
    # Titans/MIRAS calibration fields
    total_predictions: int = 0
    total_prediction_error: float = 0.0
    calibration_score: float = 0.5

    @property
    def score(self) -> float:
        """0-1 reputation score based on track record."""
        if self.proposals_made == 0:
            return 0.5  # Neutral for new agents
        acceptance = self.proposals_accepted / self.proposals_made
        critique_quality = (
            self.critiques_valuable / self.critiques_given
            if self.critiques_given > 0
            else 0.5
        )
        # Weight: 60% proposal acceptance, 40% critique quality
        return 0.6 * acceptance + 0.4 * critique_quality

    @property
    def vote_weight(self) -> float:
        """
        Vote weight multiplier (0.4-1.6 range).

        Includes Titans/MIRAS calibration bonus: agents with accurate
        predictions (low error) get a bonus, inaccurate ones get a penalty.
        """
        base_weight = 0.5 + self.score  # 0.5-1.5 range
        # Calibration bonus: (calibration - 0.5) * 0.2 gives -0.1 to +0.1
        calibration_bonus = (self.calibration_score - 0.5) * 0.2
        return max(0.4, min(1.6, base_weight + calibration_bonus))


class CritiqueStore:
    """
    SQLite-based storage for critique patterns.

    Enables self-improvement by:
    1. Storing successful critique -> fix patterns
    2. Retrieving similar patterns for new critiques
    3. Tracking which patterns lead to consensus
    """

    def __init__(self, db_path: str = "agora_memory.db"):
        self.db_path = Path(db_path)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection as a context manager.

        Ensures connections are properly closed even if exceptions occur.
        """
        conn = sqlite3.connect(self.db_path, timeout=DB_TIMEOUT)
        conn.execute(f"PRAGMA busy_timeout = {int(DB_TIMEOUT * 1000)}")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Debates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS debates (
                    id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    final_answer TEXT,
                    consensus_reached INTEGER,
                    confidence REAL,
                    rounds_used INTEGER,
                    duration_seconds REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Critiques table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS critiques (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    debate_id TEXT,
                    agent TEXT NOT NULL,
                    target_agent TEXT,
                    issues TEXT,  -- JSON array
                    suggestions TEXT,  -- JSON array
                    severity REAL,
                    reasoning TEXT,
                    led_to_improvement INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (debate_id) REFERENCES debates(id)
                )
            """)

            # Patterns table - aggregated successful patterns
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id TEXT PRIMARY KEY,
                    issue_type TEXT NOT NULL,
                    issue_text TEXT NOT NULL,
                    suggestion_text TEXT,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    avg_severity REAL DEFAULT 0.5,
                    example_task TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Pattern embeddings for semantic search (optional, for future)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pattern_embeddings (
                    pattern_id TEXT PRIMARY KEY,
                    embedding BLOB,
                    FOREIGN KEY (pattern_id) REFERENCES patterns(id)
                )
            """)

            # Agent reputation tracking for weighted voting
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_reputation (
                    agent_name TEXT PRIMARY KEY,
                    proposals_made INTEGER DEFAULT 0,
                    proposals_accepted INTEGER DEFAULT 0,
                    critiques_given INTEGER DEFAULT 0,
                    critiques_valuable INTEGER DEFAULT 0,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_critiques_debate ON critiques(debate_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(issue_type)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_patterns_success ON patterns(success_count DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_reputation_score ON agent_reputation(proposals_accepted DESC)"
            )

            # === Titans/MIRAS Migrations ===
            # Add new columns for surprise-based learning and prediction tracking
            # These use try/except to be idempotent (safe to run multiple times)

            # Patterns table: Add surprise scoring columns
            for col_def in [
                "surprise_score REAL DEFAULT 0.0",
                "base_rate REAL DEFAULT 0.5",
                "avg_prediction_error REAL DEFAULT 0.0",
                "prediction_count INTEGER DEFAULT 0",
            ]:
                try:
                    cursor.execute(f"ALTER TABLE patterns ADD COLUMN {col_def}")
                except sqlite3.OperationalError:
                    pass  # Column already exists

            # Critiques table: Add prediction tracking columns
            for col_def in [
                "expected_usefulness REAL DEFAULT 0.5",
                "actual_usefulness REAL",
                "prediction_error REAL",
            ]:
                try:
                    cursor.execute(f"ALTER TABLE critiques ADD COLUMN {col_def}")
                except sqlite3.OperationalError:
                    pass  # Column already exists

            # Agent reputation table: Add calibration scoring columns
            for col_def in [
                "total_predictions INTEGER DEFAULT 0",
                "total_prediction_error REAL DEFAULT 0.0",
                "calibration_score REAL DEFAULT 0.5",
            ]:
                try:
                    cursor.execute(f"ALTER TABLE agent_reputation ADD COLUMN {col_def}")
                except sqlite3.OperationalError:
                    pass  # Column already exists

            # Create patterns_archive table for adaptive forgetting
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patterns_archive (
                    id TEXT,
                    issue_type TEXT,
                    issue_text TEXT,
                    suggestion_text TEXT,
                    success_count INTEGER,
                    failure_count INTEGER,
                    avg_severity REAL,
                    surprise_score REAL,
                    example_task TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    archived_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

    def store_debate(self, result: DebateResult):
        """Store a complete debate result."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Store debate
            cursor.execute(
                """
                INSERT OR REPLACE INTO debates
                (id, task, final_answer, consensus_reached, confidence, rounds_used, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    result.id,
                    result.task,
                    result.final_answer,
                    1 if result.consensus_reached else 0,
                    result.confidence,
                    result.rounds_used,
                    result.duration_seconds,
                ),
            )

            # Store critiques
            for critique in result.critiques:
                cursor.execute(
                    """
                    INSERT INTO critiques
                    (debate_id, agent, target_agent, issues, suggestions, severity, reasoning)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        result.id,
                        critique.agent,
                        critique.target_agent,
                        json.dumps(critique.issues),
                        json.dumps(critique.suggestions),
                        critique.severity,
                        critique.reasoning,
                    ),
                )

            conn.commit()

    def store_pattern(self, critique: Critique, successful_fix: str):
        """Store a successful critique pattern."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            for issue in critique.issues:
                # Create pattern ID from issue hash
                pattern_id = hashlib.md5(issue.lower().encode()).hexdigest()[:12]

                # Categorize issue type (simple heuristic)
                issue_type = self._categorize_issue(issue)

                # Get matching suggestion
                suggestion = critique.suggestions[0] if critique.suggestions else ""

                # Check if pattern exists
                cursor.execute("SELECT success_count, avg_severity FROM patterns WHERE id = ?", (pattern_id,))
                existing = cursor.fetchone()

                if existing:
                    # Update existing pattern
                    new_count = existing[0] + 1
                    new_avg = (existing[1] * existing[0] + critique.severity) / new_count
                    cursor.execute(
                        """
                        UPDATE patterns
                        SET success_count = ?, avg_severity = ?, updated_at = ?
                        WHERE id = ?
                    """,
                        (new_count, new_avg, datetime.now().isoformat(), pattern_id),
                    )
                else:
                    # Insert new pattern
                    cursor.execute(
                        """
                        INSERT INTO patterns
                        (id, issue_type, issue_text, suggestion_text, success_count, avg_severity, example_task)
                        VALUES (?, ?, ?, ?, 1, ?, ?)
                    """,
                        (pattern_id, issue_type, issue, suggestion, critique.severity, successful_fix[:500]),
                    )

                # Update surprise score (Titans/MIRAS: track unexpected successes)
                self._update_surprise_score(cursor, pattern_id, is_success=True)

            conn.commit()

    def _categorize_issue(self, issue: str) -> str:
        """Simple issue categorization."""
        issue_lower = issue.lower()

        categories = {
            "performance": ["slow", "performance", "efficient", "optimize", "speed", "latency"],
            "security": ["security", "vulnerab", "injection", "auth", "permission", "xss", "csrf"],
            "correctness": ["bug", "error", "incorrect", "wrong", "fail", "break", "crash"],
            "clarity": ["unclear", "confusing", "readab", "document", "comment", "naming"],
            "architecture": ["design", "structure", "pattern", "modular", "coupling", "cohesion"],
            "completeness": ["missing", "incomplete", "todo", "edge case", "handle"],
            "testing": ["test", "coverage", "assert", "mock", "unit", "integration"],
        }

        for category, keywords in categories.items():
            if any(kw in issue_lower for kw in keywords):
                return category

        return "general"

    def fail_pattern(self, issue_text: str, issue_type: str = "general") -> None:
        """
        Record a pattern failure (critique didn't help reach consensus).

        This is the counterpart to store_pattern - called when a critique
        with matching issue text did NOT lead to improvement.
        Implements Titans/MIRAS failure tracking for balanced learning.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create pattern ID from issue hash (same as store_pattern)
            pattern_id = hashlib.md5(issue_text.lower().encode()).hexdigest()[:12]

            # Increment failure count if pattern exists
            cursor.execute(
                """
                UPDATE patterns
                SET failure_count = failure_count + 1,
                    updated_at = ?
                WHERE id = ?
                """,
                (datetime.now().isoformat(), pattern_id),
            )

            # Update surprise score based on unexpected failure
            if cursor.rowcount > 0:
                self._update_surprise_score(cursor, pattern_id, is_success=False)

            conn.commit()

    def update_prediction_outcome(
        self,
        critique_id: int,
        actual_usefulness: float,
        agent_name: Optional[str] = None,
    ) -> float:
        """
        Update critique with actual outcome, return prediction error.

        Implements Titans/MIRAS prediction error tracking - compares
        the agent's expected usefulness with the actual outcome.

        Args:
            critique_id: Database ID of the critique
            actual_usefulness: How useful the critique actually was (0.0-1.0)
            agent_name: Optional agent name to update calibration score

        Returns:
            Prediction error (|expected - actual|)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get expected usefulness
            cursor.execute(
                "SELECT expected_usefulness, agent FROM critiques WHERE id = ?",
                (critique_id,),
            )
            row = cursor.fetchone()
            if not row:
                return 0.0

            expected = row[0] if row[0] is not None else 0.5
            agent = agent_name or row[1]

            # Calculate prediction error
            prediction_error = abs(expected - actual_usefulness)

            # Update critique with outcome
            cursor.execute(
                """
                UPDATE critiques
                SET actual_usefulness = ?,
                    prediction_error = ?
                WHERE id = ?
                """,
                (actual_usefulness, prediction_error, critique_id),
            )

            # Update agent's calibration score if agent provided
            if agent:
                self._update_agent_calibration(cursor, agent, prediction_error)

            conn.commit()
            return prediction_error

    def _update_agent_calibration(
        self,
        cursor,
        agent_name: str,
        prediction_error: float,
    ) -> None:
        """
        Update agent's calibration score based on prediction accuracy.

        Agents with lower average prediction error get better calibration scores.
        """
        # Ensure agent exists
        cursor.execute(
            "INSERT OR IGNORE INTO agent_reputation (agent_name) VALUES (?)",
            (agent_name,),
        )

        # Update prediction tracking and calibration
        cursor.execute(
            """
            UPDATE agent_reputation
            SET total_predictions = total_predictions + 1,
                total_prediction_error = total_prediction_error + ?,
                calibration_score = 1.0 - (
                    (total_prediction_error + ?) / (total_predictions + 1)
                ),
                updated_at = ?
            WHERE agent_name = ?
            """,
            (prediction_error, prediction_error, datetime.now().isoformat(), agent_name),
        )

    def _calculate_surprise(self, cursor, issue_type: str, is_success: bool) -> float:
        """
        Calculate surprise score based on deviation from base rate.

        Implements Titans/MIRAS "surprise-based memorization" - patterns
        that deviate from expected outcomes get higher surprise scores.

        Args:
            cursor: Database cursor
            issue_type: Category of the issue
            is_success: Whether this was a success (True) or failure (False)

        Returns:
            Surprise score between 0.0 and 1.0
        """
        # Get base success rate for this issue type
        cursor.execute(
            """
            SELECT AVG(
                CAST(success_count AS REAL) /
                NULLIF(success_count + failure_count, 0)
            )
            FROM patterns
            WHERE issue_type = ? AND (success_count + failure_count) > 0
            """,
            (issue_type,),
        )
        result = cursor.fetchone()
        base_rate = result[0] if result and result[0] is not None else 0.5

        # Actual outcome: 1.0 for success, 0.0 for failure
        actual = 1.0 if is_success else 0.0

        # Surprise = |actual - expected|, normalized to 0-1
        surprise = abs(actual - base_rate)
        return min(1.0, surprise * 2)  # Scale up for visibility

    def _update_surprise_score(self, cursor, pattern_id: str, is_success: bool) -> None:
        """Update surprise score for a pattern after success/failure."""
        # Get pattern's issue_type
        cursor.execute("SELECT issue_type FROM patterns WHERE id = ?", (pattern_id,))
        result = cursor.fetchone()
        if not result:
            return

        issue_type = result[0]
        surprise = self._calculate_surprise(cursor, issue_type, is_success)

        # Update with exponential moving average (alpha = 0.3)
        cursor.execute(
            """
            UPDATE patterns
            SET surprise_score = surprise_score * 0.7 + ? * 0.3,
                base_rate = (
                    SELECT AVG(
                        CAST(success_count AS REAL) /
                        NULLIF(success_count + failure_count, 0)
                    )
                    FROM patterns WHERE issue_type = ?
                )
            WHERE id = ?
            """,
            (surprise, issue_type, pattern_id),
        )

    def retrieve_patterns(
        self,
        issue_type: Optional[str] = None,
        min_success: int = 2,
        limit: int = 10,
        decay_halflife_days: int = 30,
    ) -> list[Pattern]:
        """
        Retrieve successful patterns with Titans/MIRAS-inspired ranking.

        Ranking formula:
            score = (success_count * (1 + surprise_score)) /
                    (1 + age_days / decay_halflife_days)

        This prioritizes:
        - Higher success counts
        - More surprising patterns (unexpected successes)
        - Recent patterns (time-decay)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build query with Titans/MIRAS-inspired ranking
            base_sql = """
                SELECT id, issue_type, issue_text, suggestion_text, success_count,
                       failure_count, avg_severity, example_task, created_at, updated_at,
                       (success_count * (1 + COALESCE(surprise_score, 0))) /
                       (1 + (julianday('now') - julianday(updated_at)) / ?) as decay_score
                FROM patterns
                WHERE success_count >= ?
            """
            params = [decay_halflife_days, min_success]

            if issue_type:
                base_sql += " AND issue_type = ?"
                params.append(issue_type)

            base_sql += " ORDER BY decay_score DESC LIMIT ?"
            params.append(limit)

            cursor.execute(base_sql, params)

            patterns = [
                Pattern(
                    id=row[0],
                    issue_type=row[1],
                    issue_text=row[2],
                    suggestion_text=row[3],
                    success_count=row[4],
                    failure_count=row[5],
                    avg_severity=row[6],
                    example_task=row[7],
                    created_at=row[8],
                    updated_at=row[9],
                )
                for row in cursor.fetchall()
            ]

            return patterns

    def get_stats(self) -> dict:
        """Get statistics about stored patterns and debates."""
        # Ensure tables exist
        self._init_db()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            cursor.execute("SELECT COUNT(*) FROM debates")
            stats["total_debates"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM debates WHERE consensus_reached = 1")
            stats["consensus_debates"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM critiques")
            stats["total_critiques"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM patterns")
            stats["total_patterns"] = cursor.fetchone()[0]

            cursor.execute("SELECT issue_type, COUNT(*) FROM patterns GROUP BY issue_type")
            stats["patterns_by_type"] = dict(cursor.fetchall())

            cursor.execute("SELECT AVG(confidence) FROM debates WHERE consensus_reached = 1")
            avg_conf = cursor.fetchone()[0]
            stats["avg_consensus_confidence"] = avg_conf if avg_conf else 0.0

            return stats

    def export_for_training(self) -> list[dict]:
        """Export successful patterns for potential fine-tuning."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT d.task, c.issues, c.suggestions, d.final_answer, d.consensus_reached
                FROM critiques c
                JOIN debates d ON c.debate_id = d.id
                WHERE d.consensus_reached = 1
            """
            )

            training_data = []
            for row in cursor.fetchall():
                training_data.append(
                    {
                        "task": row[0],
                        "issues": json.loads(row[1]) if row[1] else [],
                        "suggestions": json.loads(row[2]) if row[2] else [],
                        "successful_answer": row[3],
                    }
                )

            return training_data

    # =========================================================================
    # Agent Reputation Tracking
    # =========================================================================

    def get_reputation(self, agent_name: str) -> Optional[AgentReputation]:
        """Get reputation for an agent."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT agent_name, proposals_made, proposals_accepted,
                       critiques_given, critiques_valuable, updated_at,
                       COALESCE(total_predictions, 0),
                       COALESCE(total_prediction_error, 0.0),
                       COALESCE(calibration_score, 0.5)
                FROM agent_reputation
                WHERE agent_name = ?
            """,
                (agent_name,),
            )
            row = cursor.fetchone()

            if not row:
                return None

            return AgentReputation(
                agent_name=row[0],
                proposals_made=row[1],
                proposals_accepted=row[2],
                critiques_given=row[3],
                critiques_valuable=row[4],
                updated_at=row[5],
                total_predictions=row[6],
                total_prediction_error=row[7],
                calibration_score=row[8],
            )

    def get_vote_weight(self, agent_name: str) -> float:
        """Get vote weight for an agent (0.5-1.5 range based on reputation)."""
        rep = self.get_reputation(agent_name)
        if not rep:
            return 1.0  # Neutral weight for unknown agents
        return rep.vote_weight

    def update_reputation(
        self,
        agent_name: str,
        proposal_made: bool = False,
        proposal_accepted: bool = False,
        critique_given: bool = False,
        critique_valuable: bool = False,
    ) -> None:
        """Update reputation metrics for an agent."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Ensure agent exists
            cursor.execute(
                """
                INSERT OR IGNORE INTO agent_reputation (agent_name)
                VALUES (?)
            """,
                (agent_name,),
            )

            # Update metrics
            updates = []
            if proposal_made:
                updates.append("proposals_made = proposals_made + 1")
            if proposal_accepted:
                updates.append("proposals_accepted = proposals_accepted + 1")
            if critique_given:
                updates.append("critiques_given = critiques_given + 1")
            if critique_valuable:
                updates.append("critiques_valuable = critiques_valuable + 1")

            if updates:
                updates.append(f"updated_at = '{datetime.now().isoformat()}'")
                cursor.execute(
                    f"""
                    UPDATE agent_reputation
                    SET {', '.join(updates)}
                    WHERE agent_name = ?
                """,
                    (agent_name,),
                )

            conn.commit()

    def get_all_reputations(self) -> list[AgentReputation]:
        """Get all agent reputations, ordered by score."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT agent_name, proposals_made, proposals_accepted,
                       critiques_given, critiques_valuable, updated_at,
                       COALESCE(total_predictions, 0),
                       COALESCE(total_prediction_error, 0.0),
                       COALESCE(calibration_score, 0.5)
                FROM agent_reputation
                ORDER BY proposals_accepted DESC
            """
            )

            reputations = [
                AgentReputation(
                    agent_name=row[0],
                    proposals_made=row[1],
                    proposals_accepted=row[2],
                    critiques_given=row[3],
                    critiques_valuable=row[4],
                    updated_at=row[5],
                    total_predictions=row[6],
                    total_prediction_error=row[7],
                    calibration_score=row[8],
                )
                for row in cursor.fetchall()
            ]

            return reputations

    # =========================================================================
    # Adaptive Forgetting (Titans/MIRAS)
    # =========================================================================

    def prune_stale_patterns(
        self,
        max_age_days: int = 90,
        min_success_rate: float = 0.3,
        archive: bool = True,
    ) -> int:
        """
        Remove or archive patterns that are stale or unsuccessful.

        Implements Titans/MIRAS "adaptive forgetting" - discards obsolete
        information to prevent memory bloat and outdated patterns from
        interfering with learning.

        Args:
            max_age_days: Patterns older than this without updates get pruned
            min_success_rate: Patterns below this success rate get pruned
            archive: If True, move to archive table instead of deleting

        Returns:
            Number of patterns pruned
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if archive:
            # Move stale/unsuccessful patterns to archive table
            cursor.execute(
                """
                INSERT INTO patterns_archive
                    (id, issue_type, issue_text, suggestion_text, success_count,
                     failure_count, avg_severity, surprise_score, example_task,
                     created_at, updated_at)
                SELECT id, issue_type, issue_text, suggestion_text, success_count,
                       failure_count, avg_severity, surprise_score, example_task,
                       created_at, updated_at
                FROM patterns
                WHERE julianday('now') - julianday(updated_at) > ?
                  AND (
                    CAST(success_count AS REAL) /
                    NULLIF(success_count + failure_count, 0)
                  ) < ?
                """,
                (max_age_days, min_success_rate),
            )

        # Delete stale/unsuccessful patterns
        cursor.execute(
            """
            DELETE FROM patterns
            WHERE julianday('now') - julianday(updated_at) > ?
              AND (
                CAST(success_count AS REAL) /
                NULLIF(success_count + failure_count, 0)
              ) < ?
            """,
            (max_age_days, min_success_rate),
        )

        pruned = cursor.rowcount
        conn.commit()
        conn.close()
        return pruned

    def get_archive_stats(self) -> dict:
        """Get statistics about archived patterns."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM patterns_archive")
        total = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT issue_type, COUNT(*)
            FROM patterns_archive
            GROUP BY issue_type
            """
        )
        by_type = dict(cursor.fetchall())

        conn.close()
        return {"total_archived": total, "archived_by_type": by_type}
