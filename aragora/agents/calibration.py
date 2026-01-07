"""
CalibrationTracker - Track prediction accuracy for agent calibration.

Records prediction confidence vs actual outcomes to compute:
- Brier scores (mean squared error of predictions)
- Expected Calibration Error (ECE)
- Calibration curves per agent and domain

Well-calibrated agents have confidence that matches their accuracy:
- 80% confidence predictions should be correct 80% of the time
"""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from aragora.config import DB_CALIBRATION_PATH
from aragora.storage.schema import SchemaManager

# Database connection timeout in seconds
DB_TIMEOUT_SECONDS = 30

# Schema version for CalibrationTracker migrations
CALIBRATION_SCHEMA_VERSION = 1


@dataclass
class CalibrationBucket:
    """Stats for a confidence range (e.g., 0.7-0.8)."""

    range_start: float
    range_end: float
    total_predictions: int = 0
    correct_predictions: int = 0
    brier_sum: float = 0.0

    @property
    def accuracy(self) -> float:
        """Actual accuracy in this bucket."""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions

    @property
    def expected_accuracy(self) -> float:
        """Expected accuracy (midpoint of bucket)."""
        return (self.range_start + self.range_end) / 2

    @property
    def calibration_error(self) -> float:
        """Absolute difference between expected and actual accuracy."""
        return abs(self.accuracy - self.expected_accuracy)

    @property
    def brier_score(self) -> float:
        """Average Brier score for this bucket."""
        if self.total_predictions == 0:
            return 0.0
        return self.brier_sum / self.total_predictions


@dataclass
class CalibrationSummary:
    """Summary of an agent's calibration performance."""

    agent: str
    total_predictions: int = 0
    total_correct: int = 0
    brier_score: float = 0.0
    ece: float = 0.0  # Expected Calibration Error
    buckets: list[CalibrationBucket] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Overall prediction accuracy."""
        if self.total_predictions == 0:
            return 0.0
        return self.total_correct / self.total_predictions

    @property
    def is_overconfident(self) -> bool:
        """True if agent's confidence exceeds accuracy."""
        if not self.buckets:
            return False
        high_conf = [b for b in self.buckets if b.range_start >= 0.7]
        if not high_conf:
            return False
        active = [b for b in high_conf if b.total_predictions > 0]
        if not active:
            return False
        avg_error = sum(b.expected_accuracy - b.accuracy for b in active) / len(active)
        return avg_error > 0.1

    @property
    def is_underconfident(self) -> bool:
        """True if agent's accuracy exceeds confidence."""
        if not self.buckets:
            return False
        low_conf = [b for b in self.buckets if b.range_end <= 0.5]
        if not low_conf:
            return False
        active = [b for b in low_conf if b.total_predictions > 0]
        if not active:
            return False
        avg_error = sum(b.accuracy - b.expected_accuracy for b in active) / len(active)
        return avg_error > 0.1


class CalibrationTracker:
    """
    Track prediction calibration for agents.

    Records confidence → outcome pairs and computes calibration metrics.
    Stores data in SQLite for persistence across sessions.
    """

    def __init__(self, db_path: str = DB_CALIBRATION_PATH):
        self.db_path = Path(db_path)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with guaranteed cleanup."""
        conn = sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database tables using SchemaManager."""
        with self._get_connection() as conn:
            # Use SchemaManager for version tracking and migrations
            manager = SchemaManager(
                conn, "calibration", current_version=CALIBRATION_SCHEMA_VERSION
            )

            # Initial schema (v1)
            initial_schema = """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    correct INTEGER NOT NULL,
                    domain TEXT DEFAULT 'general',
                    debate_id TEXT,
                    position_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_pred_agent ON predictions(agent);
                CREATE INDEX IF NOT EXISTS idx_pred_domain ON predictions(domain);
                CREATE INDEX IF NOT EXISTS idx_pred_confidence ON predictions(confidence);
            """

            manager.ensure_schema(initial_schema=initial_schema)

    def record_prediction(
        self,
        agent: str,
        confidence: float,
        correct: bool,
        domain: str = "general",
        debate_id: str = "",
        position_id: str = "",
    ) -> int:
        """
        Record a prediction and its outcome.

        Args:
            agent: Agent name
            confidence: Expressed confidence (0.0-1.0)
            correct: Whether the prediction was correct
            domain: Problem domain (e.g., "security", "performance")
            debate_id: Optional debate reference
            position_id: Optional position reference

        Returns:
            ID of the recorded prediction
        """
        confidence = max(0.0, min(1.0, confidence))

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO predictions (agent, confidence, correct, domain, debate_id, position_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agent,
                    confidence,
                    1 if correct else 0,
                    domain,
                    debate_id,
                    position_id,
                    datetime.now().isoformat(),
                ),
            )
            pred_id = cursor.lastrowid
            conn.commit()

        return pred_id or 0

    def get_calibration_curve(
        self,
        agent: str,
        num_buckets: int = 10,
        domain: Optional[str] = None,
    ) -> list[CalibrationBucket]:
        """
        Get calibration curve (expected vs actual accuracy per bucket).

        Args:
            agent: Agent name
            num_buckets: Number of confidence buckets (default 10)
            domain: Optional domain filter

        Returns:
            List of CalibrationBucket objects ordered by confidence range
        """
        bucket_size = 1.0 / num_buckets
        buckets = []

        with self._get_connection() as conn:
            for i in range(num_buckets):
                range_start = i * bucket_size
                range_end = (i + 1) * bucket_size

                # Last bucket uses <= to include 1.0 exactly
                if i == num_buckets - 1:
                    query = """
                        SELECT COUNT(*), SUM(correct)
                        FROM predictions
                        WHERE agent = ? AND confidence >= ? AND confidence <= ?
                    """
                else:
                    query = """
                        SELECT COUNT(*), SUM(correct)
                        FROM predictions
                        WHERE agent = ? AND confidence >= ? AND confidence < ?
                    """
                params: list = [agent, range_start, range_end]

                if domain:
                    query += " AND domain = ?"
                    params.append(domain)

                cursor = conn.execute(query, params)
                row = cursor.fetchone()

                total = row[0] or 0
                correct = row[1] or 0

                # Compute Brier sum for this bucket (last bucket uses <= to include 1.0)
                if i == num_buckets - 1:
                    brier_query = """
                        SELECT SUM((confidence - correct) * (confidence - correct))
                        FROM predictions
                        WHERE agent = ? AND confidence >= ? AND confidence <= ?
                    """
                else:
                    brier_query = """
                        SELECT SUM((confidence - correct) * (confidence - correct))
                        FROM predictions
                        WHERE agent = ? AND confidence >= ? AND confidence < ?
                    """
                if domain:
                    brier_query += " AND domain = ?"

                cursor = conn.execute(brier_query, params)
                brier_row = cursor.fetchone()
                brier_sum = brier_row[0] or 0.0

                buckets.append(
                    CalibrationBucket(
                        range_start=range_start,
                        range_end=range_end,
                        total_predictions=total,
                        correct_predictions=correct,
                        brier_sum=brier_sum,
                    )
                )

        return buckets

    def get_brier_score(self, agent: str, domain: Optional[str] = None) -> float:
        """
        Compute Brier score for an agent.

        Brier score = mean((confidence - outcome)^2)
        Lower is better. 0 = perfect, 0.25 = random at 50% confidence.

        Args:
            agent: Agent name
            domain: Optional domain filter

        Returns:
            Brier score (0.0 to 1.0)
        """
        with self._get_connection() as conn:
            query = """
                SELECT AVG((confidence - correct) * (confidence - correct))
                FROM predictions
                WHERE agent = ?
            """
            params: list = [agent]

            if domain:
                query += " AND domain = ?"
                params.append(domain)

            cursor = conn.execute(query, params)
            row = cursor.fetchone()

        return row[0] if row[0] is not None else 0.0

    def get_expected_calibration_error(
        self,
        agent: str,
        num_buckets: int = 10,
        domain: Optional[str] = None,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        ECE = weighted average of |accuracy - confidence| per bucket,
        weighted by number of predictions in each bucket.

        Args:
            agent: Agent name
            num_buckets: Number of confidence buckets
            domain: Optional domain filter

        Returns:
            ECE (0.0 to 1.0, lower is better)
        """
        buckets = self.get_calibration_curve(agent, num_buckets, domain)

        total_predictions = sum(b.total_predictions for b in buckets)
        if total_predictions == 0:
            return 0.0

        ece = sum(
            (b.total_predictions / total_predictions) * b.calibration_error
            for b in buckets
            if b.total_predictions > 0
        )

        return ece

    def get_calibration_summary(
        self,
        agent: str,
        domain: Optional[str] = None,
    ) -> CalibrationSummary:
        """
        Get comprehensive calibration summary for an agent.

        Args:
            agent: Agent name
            domain: Optional domain filter

        Returns:
            CalibrationSummary with all metrics
        """
        buckets = self.get_calibration_curve(agent, domain=domain)

        total_predictions = sum(b.total_predictions for b in buckets)
        total_correct = sum(b.correct_predictions for b in buckets)
        brier_score = self.get_brier_score(agent, domain)
        ece = self.get_expected_calibration_error(agent, domain=domain)

        return CalibrationSummary(
            agent=agent,
            total_predictions=total_predictions,
            total_correct=total_correct,
            brier_score=brier_score,
            ece=ece,
            buckets=buckets,
        )

    def get_domain_breakdown(self, agent: str) -> dict[str, CalibrationSummary]:
        """
        Get calibration breakdown by domain for an agent.

        Returns:
            Dict mapping domain name to CalibrationSummary
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT domain FROM predictions WHERE agent = ?",
                (agent,),
            )
            domains = [row[0] for row in cursor.fetchall()]

        return {domain: self.get_calibration_summary(agent, domain) for domain in domains}

    def get_all_agents(self) -> list[str]:
        """Get list of all agents with recorded predictions."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT DISTINCT agent FROM predictions ORDER BY agent")
            agents = [row[0] for row in cursor.fetchall()]
        return agents

    def get_leaderboard(
        self,
        metric: str = "brier",
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Get agents ranked by calibration metric.

        Args:
            metric: "brier" (lower is better), "ece" (lower is better),
                   or "accuracy" (higher is better)
            limit: Max number of agents to return

        Returns:
            List of (agent_name, metric_value) tuples, sorted by performance
        """
        agents = self.get_all_agents()

        results = []
        for agent in agents:
            summary = self.get_calibration_summary(agent)
            if summary.total_predictions < 5:
                continue  # Skip agents with too few predictions

            if metric == "brier":
                results.append((agent, summary.brier_score))
            elif metric == "ece":
                results.append((agent, summary.ece))
            elif metric == "accuracy":
                results.append((agent, summary.accuracy))
            else:
                results.append((agent, summary.brier_score))

        # Sort: lower is better for brier/ece, higher for accuracy
        reverse = metric == "accuracy"
        results.sort(key=lambda x: x[1], reverse=reverse)

        return results[:limit]

    def delete_agent_data(self, agent: str) -> int:
        """
        Delete all predictions for an agent.

        Returns:
            Number of records deleted
        """
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM predictions WHERE agent = ?", (agent,))
            deleted = cursor.rowcount
            conn.commit()
        return deleted


def integrate_with_position_ledger(
    calibration_tracker: CalibrationTracker,
    position_ledger,  # PositionLedger from grounded.py
    agent: str,
) -> int:
    """
    Sync resolved positions from PositionLedger to CalibrationTracker.

    Call this periodically to import position outcomes as calibration data.

    Returns:
        Number of predictions imported
    """
    positions = position_ledger.get_agent_positions(
        agent, limit=1000, outcome_filter=None
    )

    imported = 0
    for pos in positions:
        if pos.outcome in ("correct", "incorrect"):
            calibration_tracker.record_prediction(
                agent=pos.agent_name,
                confidence=pos.confidence,
                correct=(pos.outcome == "correct"),
                domain=pos.domain or "general",
                debate_id=pos.debate_id,
                position_id=pos.id,
            )
            imported += 1

    return imported
