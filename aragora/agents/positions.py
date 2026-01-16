"""
Position tracking for grounded personas.

Tracks agent positions, calibration accuracy, and provides statistical analysis
of position outcomes across debates.
"""

from __future__ import annotations

__all__ = [
    "Position",
    "CalibrationBucket",
    "DomainCalibration",
    "PositionLedger",
]

import logging
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator, Literal

from aragora.config import DB_PERSONAS_PATH
from aragora.insights.database import InsightsDatabase

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """A position/claim taken by an agent during debate."""

    id: str
    agent_name: str
    claim: str
    confidence: float  # 0.0-1.0 expressed confidence
    debate_id: str
    round_num: int
    outcome: Literal["correct", "incorrect", "unresolved", "pending"] = "pending"
    reversed: bool = False
    reversal_debate_id: str | None = None
    domain: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved_at: str | None = None

    @classmethod
    def create(
        cls,
        agent_name: str,
        claim: str,
        confidence: float,
        debate_id: str,
        round_num: int,
        domain: str | None = None,
    ) -> "Position":
        """Create a new position with generated ID."""
        return cls(
            id=str(uuid.uuid4())[:8],
            agent_name=agent_name,
            claim=claim,
            confidence=max(0.0, min(1.0, confidence)),
            debate_id=debate_id,
            round_num=round_num,
            domain=domain,
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Position":
        """Create a Position from a database row.

        Args:
            row: SQLite row with dict-like access (via row_factory = sqlite3.Row)

        Returns:
            Position instance hydrated from the row data
        """
        return cls(
            id=row["id"],
            agent_name=row["agent_name"],
            claim=row["claim"],
            confidence=row["confidence"],
            debate_id=row["debate_id"],
            round_num=row["round_num"],
            outcome=row["outcome"],
            reversed=bool(row["reversed"]),
            reversal_debate_id=row["reversal_debate_id"],
            domain=row["domain"],
            created_at=row["created_at"],
            resolved_at=row["resolved_at"],
        )


@dataclass
class CalibrationBucket:
    """Calibration statistics for a confidence range."""

    bucket_start: float  # e.g., 0.8
    bucket_end: float  # e.g., 0.9
    predictions: int = 0
    correct: int = 0
    brier_sum: float = 0.0

    @property
    def accuracy(self) -> float:
        """Actual accuracy in this bucket."""
        return self.correct / self.predictions if self.predictions > 0 else 0.0

    @property
    def expected_accuracy(self) -> float:
        """Expected accuracy (midpoint of bucket)."""
        return (self.bucket_start + self.bucket_end) / 2

    @property
    def calibration_error(self) -> float:
        """Expected calibration error for this bucket."""
        return abs(self.accuracy - self.expected_accuracy)

    @property
    def bucket_key(self) -> str:
        """Key for storage (e.g., '0.8-0.9')."""
        return f"{self.bucket_start:.1f}-{self.bucket_end:.1f}"


@dataclass
class DomainCalibration:
    """Per-domain calibration tracking."""

    domain: str
    total_predictions: int = 0
    total_correct: int = 0
    brier_sum: float = 0.0
    buckets: dict[str, CalibrationBucket] = field(default_factory=dict)

    @property
    def calibration_score(self) -> float:
        """1 - avg_brier_score (higher is better)."""
        if self.total_predictions == 0:
            return 0.5
        return 1.0 - (self.brier_sum / self.total_predictions)

    @property
    def accuracy(self) -> float:
        """Overall accuracy in this domain."""
        if self.total_predictions == 0:
            return 0.0
        return self.total_correct / self.total_predictions


class PositionLedger:
    """
    Tracks every position an agent takes across debates.

    Integrates with PersonaManager's database for unified agent data.
    """

    def __init__(self, db_path: str = DB_PERSONAS_PATH):
        self.db_path = Path(db_path)
        self.db = InsightsDatabase(db_path)
        self._init_tables()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with guaranteed cleanup."""
        with self.db.connection() as conn:
            yield conn

    def _init_tables(self) -> None:
        """Add positions table if not exists."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    claim TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    debate_id TEXT NOT NULL,
                    round_num INTEGER NOT NULL,
                    outcome TEXT DEFAULT 'pending',
                    reversed INTEGER DEFAULT 0,
                    reversal_debate_id TEXT,
                    domain TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TEXT
                )
            """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_agent ON positions(agent_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_debate ON positions(debate_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_outcome ON positions(outcome)")
            conn.commit()

    def record_position(
        self,
        agent_name: str,
        claim: str,
        confidence: float,
        debate_id: str,
        round_num: int,
        domain: str | None = None,
    ) -> str:
        """Record a new position. Returns position ID."""
        position = Position.create(
            agent_name=agent_name,
            claim=claim,
            confidence=confidence,
            debate_id=debate_id,
            round_num=round_num,
            domain=domain,
        )

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO positions
                (id, agent_name, claim, confidence, debate_id, round_num, domain, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    position.id,
                    position.agent_name,
                    position.claim,
                    position.confidence,
                    position.debate_id,
                    position.round_num,
                    position.domain,
                    position.created_at,
                ),
            )
            conn.commit()

        return position.id

    def resolve_position(
        self,
        position_id: str,
        outcome: Literal["correct", "incorrect", "unresolved"],
    ) -> None:
        """Mark a position's outcome after debate conclusion."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE positions
                SET outcome = ?, resolved_at = ?
                WHERE id = ?
                """,
                (outcome, datetime.now().isoformat(), position_id),
            )
            conn.commit()

    def record_reversal(
        self,
        agent_name: str,
        original_position_id: str,
        new_debate_id: str,
    ) -> None:
        """Record when agent reverses a previous position."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE positions
                SET reversed = 1, reversal_debate_id = ?
                WHERE id = ? AND agent_name = ?
                """,
                (new_debate_id, original_position_id, agent_name),
            )
            conn.commit()

    def get_agent_positions(
        self,
        agent_name: str,
        limit: int = 100,
        outcome_filter: str | None = None,
    ) -> list[Position]:
        """Get positions for an agent."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM positions WHERE agent_name = ?"
            params: list = [agent_name]

            if outcome_filter:
                query += " AND outcome = ?"
                params.append(outcome_filter)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return [Position.from_row(row) for row in rows]

    def get_position_stats(self, agent_name: str) -> dict:
        """Get aggregate position statistics."""
        with self._get_connection() as conn:
            # Total positions
            cursor = conn.execute(
                "SELECT COUNT(*) FROM positions WHERE agent_name = ?", (agent_name,)
            )
            row = cursor.fetchone()
            total = row[0] if row else 0

            # By outcome
            cursor = conn.execute(
                """
                SELECT outcome, COUNT(*)
                FROM positions
                WHERE agent_name = ?
                GROUP BY outcome
                """,
                (agent_name,),
            )
            outcomes = dict(cursor.fetchall())

            # Reversals
            cursor = conn.execute(
                "SELECT COUNT(*) FROM positions WHERE agent_name = ? AND reversed = 1",
                (agent_name,),
            )
            row = cursor.fetchone()
            reversals = row[0] if row else 0

            # Average confidence by outcome
            cursor = conn.execute(
                """
                SELECT outcome, AVG(confidence)
                FROM positions
                WHERE agent_name = ? AND outcome != 'pending'
                GROUP BY outcome
                """,
                (agent_name,),
            )
            avg_confidence = dict(cursor.fetchall())

        return {
            "total": total,
            "correct": outcomes.get("correct", 0),
            "incorrect": outcomes.get("incorrect", 0),
            "unresolved": outcomes.get("unresolved", 0),
            "pending": outcomes.get("pending", 0),
            "reversals": reversals,
            "avg_confidence_when_correct": avg_confidence.get("correct", 0),
            "avg_confidence_when_incorrect": avg_confidence.get("incorrect", 0),
        }

    def get_positions_for_debate(self, debate_id: str) -> list[Position]:
        """Get all positions from a specific debate."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute(
                "SELECT * FROM positions WHERE debate_id = ? ORDER BY round_num, agent_name",
                (debate_id,),
            )
            rows = cursor.fetchall()

        return [Position.from_row(row) for row in rows]

    def detect_domain(self, content: str) -> str | None:
        """Detect expertise domain from content using keywords."""
        content_lower = content.lower()

        domain_keywords = {
            "security": ["security", "vulnerability", "auth", "xss", "injection", "csrf"],
            "performance": ["performance", "optimization", "latency", "throughput", "cache"],
            "architecture": ["architecture", "design pattern", "modularity", "coupling"],
            "testing": ["test", "coverage", "mock", "fixture", "assertion"],
            "error_handling": ["error", "exception", "fallback", "retry", "graceful"],
            "concurrency": ["async", "parallel", "thread", "race condition", "deadlock"],
            "api_design": ["api", "endpoint", "rest", "graphql", "interface"],
            "database": ["database", "query", "index", "transaction", "schema"],
            "frontend": ["ui", "component", "render", "css", "responsive"],
            "devops": ["deploy", "ci/cd", "docker", "kubernetes", "infrastructure"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in content_lower for kw in keywords):
                return domain

        return None
