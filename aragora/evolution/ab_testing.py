"""
A/B testing framework for prompt evolution.

Enables scientific comparison of evolved prompts vs baseline prompts
through controlled debate experiments.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from aragora.config import DB_TIMEOUT_SECONDS, resolve_db_path
from aragora.storage.base_store import SQLiteStore

logger = logging.getLogger(__name__)

# Explicit column list for SELECT queries - must match ABTest.from_row() order
AB_TEST_COLUMNS = """id, agent, baseline_prompt_version, evolved_prompt_version,
    baseline_wins, evolved_wins, baseline_debates, evolved_debates,
    started_at, concluded_at, status, metadata"""


class ABTestStatus(Enum):
    """Status of an A/B test."""

    ACTIVE = "active"
    CONCLUDED = "concluded"
    CANCELLED = "cancelled"


@dataclass
class ABTest:
    """
    An A/B test comparing baseline and evolved prompts.

    Tracks wins and losses for each variant to determine
    if the evolved prompt is actually an improvement.
    """

    id: str
    agent: str
    baseline_prompt_version: int
    evolved_prompt_version: int
    baseline_wins: int = 0
    evolved_wins: int = 0
    baseline_debates: int = 0
    evolved_debates: int = 0
    started_at: str = ""
    concluded_at: Optional[str] = None
    status: ABTestStatus = ABTestStatus.ACTIVE
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.utcnow().isoformat()

    @property
    def evolved_win_rate(self) -> float:
        """Calculate win rate for evolved prompt."""
        total = self.baseline_wins + self.evolved_wins
        return self.evolved_wins / total if total > 0 else 0.5

    @property
    def baseline_win_rate(self) -> float:
        """Calculate win rate for baseline prompt."""
        total = self.baseline_wins + self.evolved_wins
        return self.baseline_wins / total if total > 0 else 0.5

    @property
    def total_debates(self) -> int:
        """Total debates in the test."""
        return self.baseline_debates + self.evolved_debates

    @property
    def sample_size(self) -> int:
        """Number of decided debates (with a winner)."""
        return self.baseline_wins + self.evolved_wins

    @property
    def is_significant(self) -> bool:
        """
        Check if results are statistically significant.

        Uses a simple threshold: at least 20 samples and
        > 60% win rate difference.
        """
        if self.sample_size < 20:
            return False

        diff = abs(self.evolved_win_rate - 0.5)
        return diff > 0.1  # 10% improvement threshold

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "agent": self.agent,
            "baseline_prompt_version": self.baseline_prompt_version,
            "evolved_prompt_version": self.evolved_prompt_version,
            "baseline_wins": self.baseline_wins,
            "evolved_wins": self.evolved_wins,
            "baseline_debates": self.baseline_debates,
            "evolved_debates": self.evolved_debates,
            "evolved_win_rate": self.evolved_win_rate,
            "baseline_win_rate": self.baseline_win_rate,
            "total_debates": self.total_debates,
            "sample_size": self.sample_size,
            "is_significant": self.is_significant,
            "started_at": self.started_at,
            "concluded_at": self.concluded_at,
            "status": self.status.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: tuple) -> "ABTest":
        """Create from database row."""
        return cls(
            id=row[0],
            agent=row[1],
            baseline_prompt_version=row[2],
            evolved_prompt_version=row[3],
            baseline_wins=row[4],
            evolved_wins=row[5],
            baseline_debates=row[6],
            evolved_debates=row[7],
            started_at=row[8],
            concluded_at=row[9],
            status=ABTestStatus(row[10]) if row[10] else ABTestStatus.ACTIVE,
            metadata=json.loads(row[11]) if row[11] else {},
        )


@dataclass
class ABTestResult:
    """Result of concluding an A/B test."""

    test_id: str
    winner: str  # "baseline", "evolved", or "tie"
    confidence: float
    recommendation: str
    stats: dict = field(default_factory=dict)


class ABTestManager(SQLiteStore):
    """
    Manages A/B tests for prompt evolution.

    Provides:
    - Test creation and lifecycle management
    - Result recording and aggregation
    - Statistical analysis for decision making

    Inherits from SQLiteStore for standardized schema management.
    """

    SCHEMA_NAME = "ab_testing"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS ab_tests (
            id TEXT PRIMARY KEY,
            agent TEXT NOT NULL,
            baseline_prompt_version INTEGER NOT NULL,
            evolved_prompt_version INTEGER NOT NULL,
            baseline_wins INTEGER DEFAULT 0,
            evolved_wins INTEGER DEFAULT 0,
            baseline_debates INTEGER DEFAULT 0,
            evolved_debates INTEGER DEFAULT 0,
            started_at TEXT,
            concluded_at TEXT,
            status TEXT DEFAULT 'active',
            metadata TEXT,
            UNIQUE(agent, status)
        );

        CREATE TABLE IF NOT EXISTS ab_test_debates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_id TEXT NOT NULL,
            debate_id TEXT NOT NULL,
            variant TEXT NOT NULL,
            won INTEGER NOT NULL,
            recorded_at TEXT,
            FOREIGN KEY (test_id) REFERENCES ab_tests(id),
            UNIQUE(test_id, debate_id)
        );
    """

    def __init__(self, db_path: str = "ab_tests.db"):
        """
        Initialize the A/B test manager.

        Args:
            db_path: Path to SQLite database file
        """
        super().__init__(resolve_db_path(db_path), timeout=DB_TIMEOUT_SECONDS)

    def start_test(
        self,
        agent: str,
        baseline_version: int,
        evolved_version: int,
        metadata: Optional[dict] = None,
    ) -> ABTest:
        """
        Start a new A/B test for an agent.

        Args:
            agent: Agent name
            baseline_version: Version number of baseline prompt
            evolved_version: Version number of evolved prompt
            metadata: Optional additional metadata

        Returns:
            The created ABTest

        Raises:
            ValueError: If agent already has an active test
        """
        # Check for existing active test
        existing = self.get_active_test(agent)
        if existing:
            raise ValueError(f"Agent {agent} already has an active test: {existing.id}")

        test = ABTest(
            id=str(uuid.uuid4()),
            agent=agent,
            baseline_prompt_version=baseline_version,
            evolved_prompt_version=evolved_version,
            metadata=metadata or {},
        )

        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO ab_tests
                (id, agent, baseline_prompt_version, evolved_prompt_version,
                 started_at, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    test.id,
                    test.agent,
                    test.baseline_prompt_version,
                    test.evolved_prompt_version,
                    test.started_at,
                    test.status.value,
                    json.dumps(test.metadata),
                ),
            )

        logger.info(
            f"Started A/B test {test.id} for {agent}: " f"v{baseline_version} vs v{evolved_version}"
        )

        return test

    def get_test(self, test_id: str) -> Optional[ABTest]:
        """Get a specific A/B test by ID."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT {AB_TEST_COLUMNS} FROM ab_tests WHERE id = ?",
                (test_id,),
            )
            row = cursor.fetchone()

            if row:
                return ABTest.from_row(row)
            return None

    def get_active_test(self, agent: str) -> Optional[ABTest]:
        """Get the active A/B test for an agent, if any."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT {AB_TEST_COLUMNS} FROM ab_tests WHERE agent = ? AND status = 'active'",
                (agent,),
            )
            row = cursor.fetchone()

            if row:
                return ABTest.from_row(row)
            return None

    def get_agent_tests(self, agent: str, limit: int = 10) -> list[ABTest]:
        """Get all tests for an agent."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT {AB_TEST_COLUMNS} FROM ab_tests
                WHERE agent = ?
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (agent, limit),
            )

            return [ABTest.from_row(row) for row in cursor.fetchall()]

    def record_result(
        self,
        agent: str,
        debate_id: str,
        variant: str,
        won: bool,
    ) -> Optional[ABTest]:
        """
        Record a debate result for the active test.

        Args:
            agent: Agent name
            debate_id: ID of the debate
            variant: Which variant was used ("baseline" or "evolved")
            won: Whether the agent won the debate

        Returns:
            Updated ABTest or None if no active test
        """
        test = self.get_active_test(agent)
        if not test:
            logger.debug(f"No active A/B test for {agent}")
            return None

        if variant not in ("baseline", "evolved"):
            raise ValueError(f"Invalid variant: {variant}")

        with self.connection() as conn:
            cursor = conn.cursor()

            # Record the debate
            try:
                cursor.execute(
                    """
                    INSERT INTO ab_test_debates
                    (test_id, debate_id, variant, won, recorded_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        test.id,
                        debate_id,
                        variant,
                        1 if won else 0,
                        datetime.utcnow().isoformat(),
                    ),
                )
            except sqlite3.IntegrityError:
                logger.warning(f"Debate {debate_id} already recorded for test {test.id}")
                return test

            # Update test counters
            if variant == "baseline":
                cursor.execute(
                    """
                    UPDATE ab_tests
                    SET baseline_debates = baseline_debates + 1,
                        baseline_wins = baseline_wins + ?
                    WHERE id = ?
                    """,
                    (1 if won else 0, test.id),
                )
            else:
                cursor.execute(
                    """
                    UPDATE ab_tests
                    SET evolved_debates = evolved_debates + 1,
                        evolved_wins = evolved_wins + ?
                    WHERE id = ?
                    """,
                    (1 if won else 0, test.id),
                )

        logger.info(f"Recorded {variant} {'win' if won else 'loss'} for test {test.id}")

        return self.get_test(test.id)

    def conclude_test(
        self,
        test_id: str,
        force: bool = False,
    ) -> ABTestResult:
        """
        Conclude an A/B test and determine the winner.

        Args:
            test_id: ID of the test to conclude
            force: Force conclusion even if not statistically significant

        Returns:
            ABTestResult with winner and recommendation
        """
        test = self.get_test(test_id)
        if not test:
            raise ValueError(f"Test not found: {test_id}")

        if test.status != ABTestStatus.ACTIVE:
            raise ValueError(f"Test already concluded: {test_id}")

        # Determine winner
        if test.sample_size == 0:
            winner = "tie"
            confidence = 0.0
            recommendation = "No data collected. Cannot determine winner."
        elif test.evolved_win_rate > 0.55:
            winner = "evolved"
            confidence = min(1.0, (test.evolved_win_rate - 0.5) * 5)
            recommendation = (
                f"Evolved prompt (v{test.evolved_prompt_version}) wins with "
                f"{test.evolved_win_rate:.1%} win rate. Recommend adoption."
            )
        elif test.baseline_win_rate > 0.55:
            winner = "baseline"
            confidence = min(1.0, (test.baseline_win_rate - 0.5) * 5)
            recommendation = (
                f"Baseline prompt (v{test.baseline_prompt_version}) performs better. "
                f"Recommend keeping current version."
            )
        else:
            winner = "tie"
            confidence = 0.3
            recommendation = (
                "No significant difference detected. "
                "Consider running longer test or trying different evolution."
            )

        # Add significance warning if needed
        if not test.is_significant and not force:
            recommendation += (
                " Note: Results may not be statistically significant "
                f"(n={test.sample_size}). Consider collecting more data."
            )

        # Update test status
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE ab_tests
                SET status = 'concluded',
                    concluded_at = ?
                WHERE id = ?
                """,
                (datetime.utcnow().isoformat(), test_id),
            )

        result = ABTestResult(
            test_id=test_id,
            winner=winner,
            confidence=confidence,
            recommendation=recommendation,
            stats={
                "evolved_win_rate": test.evolved_win_rate,
                "baseline_win_rate": test.baseline_win_rate,
                "sample_size": test.sample_size,
                "total_debates": test.total_debates,
                "is_significant": test.is_significant,
            },
        )

        logger.info(
            f"Concluded A/B test {test_id}: winner={winner}, " f"confidence={confidence:.2f}"
        )

        return result

    def cancel_test(self, test_id: str) -> bool:
        """Cancel an active test without concluding."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE ab_tests
                SET status = 'cancelled',
                    concluded_at = ?
                WHERE id = ? AND status = 'active'
                """,
                (datetime.utcnow().isoformat(), test_id),
            )
            return cursor.rowcount > 0

    def get_variant_for_debate(self, agent: str) -> Optional[str]:
        """
        Get which variant to use for the next debate.

        Alternates between baseline and evolved to ensure
        balanced sampling.

        Returns:
            "baseline" or "evolved", or None if no active test
        """
        test = self.get_active_test(agent)
        if not test:
            return None

        # Use the variant with fewer debates
        if test.baseline_debates <= test.evolved_debates:
            return "baseline"
        return "evolved"
