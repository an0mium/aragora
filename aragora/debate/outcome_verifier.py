"""Outcome Verifier — closes the feedback loop between decisions and reality.

Bridges real-world decision outcomes back into the calibration and ELO systems,
enabling empirically-grounded agent trust. This is the keystone that transforms
calibration from "how confident was the agent" into "how right was the agent."

Flow:
    Debate produces decision → OutcomeVerifier records pending decision
    → Real-world signal arrives (user feedback, test results, metric delta)
    → OutcomeVerifier.verify() feeds outcome to:
        1. CalibrationTracker (per-agent Brier scores)
        2. ELO system (agent rating adjustments)
        3. OutcomeTracker (institutional memory)
    → Next debate's TeamSelector uses updated calibration

Usage:
    verifier = OutcomeVerifier()

    # After debate completes
    verifier.record_decision(
        debate_id="d-123",
        agents=["claude", "gpt4", "gemini"],
        consensus_confidence=0.85,
        consensus_text="Deploy with canary release",
        domain="deployment",
    )

    # When ground truth arrives
    verifier.verify(
        debate_id="d-123",
        outcome_correct=True,
        signal_type="user_feedback",
        signal_detail="Canary release succeeded, promoted to 100%",
    )

    # Query systematic patterns
    patterns = verifier.get_systematic_errors(min_count=5)
    # → [{"domain": "security", "overconfidence": 0.15, "count": 12}]
"""

from __future__ import annotations

__all__ = [
    "OutcomeVerifier",
    "AsyncOutcomeVerifier",
    "PendingDecision",
    "VerificationResult",
    "SignalType",
]

import asyncio
import enum
import functools
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from aragora.config import DB_TIMEOUT_SECONDS
from aragora.persistence.db_config import get_nomic_dir
from aragora.storage.base_store import SQLiteStore

logger = logging.getLogger(__name__)

DEFAULT_VERIFIER_DB = get_nomic_dir() / "outcome_verifier.db"


class SignalType(enum.Enum):
    """Types of real-world verification signals."""

    USER_FEEDBACK = "user_feedback"
    TEST_RESULT = "test_result"
    METRIC_DELTA = "metric_delta"
    INCIDENT_REPORT = "incident_report"
    ROLLBACK = "rollback"
    MANUAL_REVIEW = "manual_review"
    AUTOMATED_CHECK = "automated_check"


@dataclass
class PendingDecision:
    """A debate decision awaiting real-world verification."""

    debate_id: str
    agents: list[str]
    consensus_confidence: float
    consensus_text: str
    domain: str = "general"
    task: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    verified: bool = False
    verified_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "debate_id": self.debate_id,
            "agents": json.dumps(self.agents),
            "consensus_confidence": self.consensus_confidence,
            "consensus_text": self.consensus_text,
            "domain": self.domain,
            "task": self.task,
            "created_at": self.created_at,
            "verified": int(self.verified),
            "verified_at": self.verified_at,
        }
        return d


@dataclass
class VerificationResult:
    """Result of verifying a decision against ground truth."""

    debate_id: str
    outcome_correct: bool
    signal_type: SignalType
    signal_detail: str
    agents_updated: list[str]
    brier_scores: dict[str, float]  # agent → individual Brier score
    calibration_adjustment: float  # system-wide adjustment recommendation
    overconfident: bool  # was this decision overconfident?
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def confidence_error(self) -> float:
        """How far off confidence was from reality (positive = overconfident)."""
        avg_brier = sum(self.brier_scores.values()) / max(len(self.brier_scores), 1)
        return avg_brier


class OutcomeVerifier:
    """Closes the feedback loop between decisions and real-world outcomes.

    Records pending decisions from debates, then accepts verification signals
    (user feedback, test results, metrics, incident reports) and feeds the
    outcomes back into the CalibrationTracker and ELO systems.

    This enables:
    1. Empirical calibration: Brier scores reflect actual decision quality
    2. Agent trust evolution: ELO ratings adjust based on real outcomes
    3. Systematic error detection: identifies domains where agents are overconfident
    4. Nomic Loop targeting: surfaces improvement goals from outcome patterns
    """

    SCHEMA_NAME = "outcome_verifier"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS pending_decisions (
            debate_id TEXT PRIMARY KEY,
            agents TEXT NOT NULL,
            consensus_confidence REAL NOT NULL,
            consensus_text TEXT,
            domain TEXT DEFAULT 'general',
            task TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            verified INTEGER DEFAULT 0,
            verified_at TEXT
        );

        CREATE TABLE IF NOT EXISTS verifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            debate_id TEXT NOT NULL,
            outcome_correct INTEGER NOT NULL,
            signal_type TEXT NOT NULL,
            signal_detail TEXT DEFAULT '',
            agents_updated TEXT NOT NULL,
            brier_scores TEXT NOT NULL,
            calibration_adjustment REAL DEFAULT 1.0,
            overconfident INTEGER DEFAULT 0,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (debate_id) REFERENCES pending_decisions(debate_id)
        );

        CREATE TABLE IF NOT EXISTS systematic_errors (
            domain TEXT NOT NULL,
            agent TEXT NOT NULL,
            total_verifications INTEGER DEFAULT 0,
            correct_count INTEGER DEFAULT 0,
            brier_sum REAL DEFAULT 0.0,
            avg_confidence REAL DEFAULT 0.0,
            last_updated TEXT,
            PRIMARY KEY (domain, agent)
        );

        CREATE INDEX IF NOT EXISTS idx_pending_unverified
        ON pending_decisions(verified) WHERE verified = 0;

        CREATE INDEX IF NOT EXISTS idx_verifications_debate
        ON verifications(debate_id);

        CREATE INDEX IF NOT EXISTS idx_systematic_domain
        ON systematic_errors(domain);
    """

    def __init__(self, db_path: Path | None = None):
        db_path = db_path or DEFAULT_VERIFIER_DB

        class _VerifierDB(SQLiteStore):
            SCHEMA_NAME = OutcomeVerifier.SCHEMA_NAME
            SCHEMA_VERSION = OutcomeVerifier.SCHEMA_VERSION
            INITIAL_SCHEMA = OutcomeVerifier.INITIAL_SCHEMA

        self.db_path = Path(db_path)
        self._db = _VerifierDB(str(db_path), timeout=DB_TIMEOUT_SECONDS)

    def record_decision(
        self,
        debate_id: str,
        agents: list[str],
        consensus_confidence: float,
        consensus_text: str,
        domain: str = "general",
        task: str = "",
    ) -> PendingDecision:
        """Record a debate decision awaiting real-world verification.

        Called automatically after debate completion. The decision sits in
        a pending state until verify() is called with ground-truth signal.
        """
        decision = PendingDecision(
            debate_id=debate_id,
            agents=agents,
            consensus_confidence=consensus_confidence,
            consensus_text=consensus_text,
            domain=domain,
            task=task,
        )

        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pending_decisions (
                    debate_id, agents, consensus_confidence, consensus_text,
                    domain, task, created_at, verified, verified_at
                ) VALUES (
                    :debate_id, :agents, :consensus_confidence, :consensus_text,
                    :domain, :task, :created_at, :verified, :verified_at
                )
                """,
                decision.to_dict(),
            )
            conn.commit()

        logger.info(
            "Recorded pending decision debate_id=%s confidence=%.2f domain=%s agents=%d",
            debate_id,
            consensus_confidence,
            domain,
            len(agents),
        )
        return decision

    def verify(
        self,
        debate_id: str,
        outcome_correct: bool,
        signal_type: SignalType | str = SignalType.USER_FEEDBACK,
        signal_detail: str = "",
    ) -> VerificationResult | None:
        """Verify a pending decision with ground-truth outcome.

        This is the core method that closes the feedback loop:
        1. Looks up the pending decision
        2. Computes Brier scores for each participating agent
        3. Feeds scores to CalibrationTracker (per-agent)
        4. Updates ELO ratings (agent-level)
        5. Records to OutcomeTracker (institutional memory)
        6. Tracks systematic errors by domain

        Args:
            debate_id: ID of the debate to verify
            outcome_correct: Whether the decision was correct
            signal_type: Type of verification signal
            signal_detail: Human-readable description of the outcome

        Returns:
            VerificationResult with detailed metrics, or None if debate not found
        """
        if isinstance(signal_type, str):
            signal_type = SignalType(signal_type)

        # 1. Look up pending decision
        decision = self._get_pending_decision(debate_id)
        if decision is None:
            logger.warning("No pending decision found for debate_id=%s", debate_id)
            return None

        if decision.verified:
            logger.info("Decision already verified for debate_id=%s", debate_id)
            return None

        agents = decision.agents
        confidence = decision.consensus_confidence
        correct_float = 1.0 if outcome_correct else 0.0

        # 2. Compute per-agent Brier scores
        brier_scores: dict[str, float] = {}
        for agent in agents:
            brier = (confidence - correct_float) ** 2
            brier_scores[agent] = brier

        overconfident = confidence > 0.7 and not outcome_correct

        # 3. Feed to CalibrationTracker
        self._update_calibration(agents, confidence, outcome_correct, decision.domain, debate_id)

        # 4. Update ELO ratings
        self._update_elo(agents, confidence, outcome_correct)

        # 5. Record to OutcomeTracker
        self._update_outcome_tracker(decision, outcome_correct, signal_detail)

        # 6. Track systematic errors
        self._update_systematic_errors(agents, decision.domain, confidence, outcome_correct)

        # 7. Mark as verified
        now = datetime.now().isoformat()
        with self._db.connection() as conn:
            conn.execute(
                "UPDATE pending_decisions SET verified = 1, verified_at = ? WHERE debate_id = ?",
                (now, debate_id),
            )
            conn.commit()

        # 8. Store verification record
        calibration_adjustment = self._compute_calibration_adjustment()
        result = VerificationResult(
            debate_id=debate_id,
            outcome_correct=outcome_correct,
            signal_type=signal_type,
            signal_detail=signal_detail,
            agents_updated=agents,
            brier_scores=brier_scores,
            calibration_adjustment=calibration_adjustment,
            overconfident=overconfident,
        )

        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO verifications (
                    debate_id, outcome_correct, signal_type, signal_detail,
                    agents_updated, brier_scores, calibration_adjustment,
                    overconfident, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    debate_id,
                    int(outcome_correct),
                    signal_type.value,
                    signal_detail,
                    json.dumps(agents),
                    json.dumps(brier_scores),
                    calibration_adjustment,
                    int(overconfident),
                    result.timestamp,
                ),
            )
            conn.commit()

        logger.info(
            "Verified debate_id=%s correct=%s overconfident=%s agents=%d",
            debate_id,
            outcome_correct,
            overconfident,
            len(agents),
        )
        return result

    def get_pending_decisions(self, limit: int = 50) -> list[PendingDecision]:
        """Get unverified pending decisions."""
        with self._db.connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM pending_decisions WHERE verified = 0 ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [self._row_to_decision(row) for row in rows]

    def get_systematic_errors(
        self,
        min_count: int = 5,
        min_overconfidence: float = 0.05,
    ) -> list[dict[str, Any]]:
        """Identify domains where agents are systematically miscalibrated.

        Returns domains where average confidence significantly exceeds
        actual success rate, indicating systematic overconfidence.

        This is the key input for Nomic Loop improvement targeting:
        systematic errors become auto-generated improvement goals.
        """
        with self._db.connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    domain,
                    agent,
                    total_verifications,
                    correct_count,
                    brier_sum,
                    avg_confidence,
                    last_updated
                FROM systematic_errors
                WHERE total_verifications >= ?
                ORDER BY brier_sum / total_verifications DESC
                """,
                (min_count,),
            ).fetchall()

        errors = []
        for row in rows:
            total = row[2]
            correct = row[3]
            brier_avg = row[4] / total if total > 0 else 0.0
            success_rate = correct / total if total > 0 else 0.0
            avg_conf = row[5]
            overconfidence = avg_conf - success_rate

            if overconfidence >= min_overconfidence:
                errors.append(
                    {
                        "domain": row[0],
                        "agent": row[1],
                        "total_verifications": total,
                        "correct_count": correct,
                        "success_rate": success_rate,
                        "avg_confidence": avg_conf,
                        "avg_brier_score": brier_avg,
                        "overconfidence": overconfidence,
                        "last_updated": row[6],
                    }
                )

        return errors

    def get_domain_calibration(self, domain: str) -> dict[str, Any]:
        """Get calibration statistics for a specific domain."""
        with self._db.connection() as conn:
            rows = conn.execute(
                """
                SELECT agent, total_verifications, correct_count, brier_sum, avg_confidence
                FROM systematic_errors
                WHERE domain = ?
                ORDER BY total_verifications DESC
                """,
                (domain,),
            ).fetchall()

        agents = {}
        for row in rows:
            total = row[1]
            agents[row[0]] = {
                "total": total,
                "correct": row[2],
                "success_rate": row[2] / total if total > 0 else 0.0,
                "avg_brier": row[3] / total if total > 0 else 0.0,
                "avg_confidence": row[4],
            }

        return {
            "domain": domain,
            "agents": agents,
            "total_verifications": sum(a["total"] for a in agents.values()),
        }

    def get_verification_history(
        self,
        limit: int = 50,
        domain: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get recent verification results."""
        with self._db.connection() as conn:
            conn.row_factory = sqlite3.Row
            if domain:
                rows = conn.execute(
                    """
                    SELECT v.*, p.domain, p.consensus_confidence
                    FROM verifications v
                    JOIN pending_decisions p ON v.debate_id = p.debate_id
                    WHERE p.domain = ?
                    ORDER BY v.timestamp DESC LIMIT ?
                    """,
                    (domain, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT v.*, p.domain, p.consensus_confidence
                    FROM verifications v
                    JOIN pending_decisions p ON v.debate_id = p.debate_id
                    ORDER BY v.timestamp DESC LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

        return [dict(row) for row in rows]

    def get_overall_stats(self) -> dict[str, Any]:
        """Get aggregate verification statistics."""
        with self._db.connection() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN outcome_correct = 1 THEN 1 ELSE 0 END) as correct,
                    SUM(CASE WHEN overconfident = 1 THEN 1 ELSE 0 END) as overconfident,
                    AVG(calibration_adjustment) as avg_adjustment
                FROM verifications
            """).fetchone()

            pending = conn.execute(
                "SELECT COUNT(*) FROM pending_decisions WHERE verified = 0"
            ).fetchone()

        total = row[0] or 0
        correct = row[1] or 0
        return {
            "total_verifications": total,
            "correct_decisions": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "overconfident_count": row[2] or 0,
            "avg_calibration_adjustment": row[3] or 1.0,
            "pending_decisions": pending[0] or 0,
        }

    # --- Internal methods ---

    def _get_pending_decision(self, debate_id: str) -> PendingDecision | None:
        with self._db.connection() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM pending_decisions WHERE debate_id = ?",
                (debate_id,),
            ).fetchone()

        if row is None:
            return None
        return self._row_to_decision(row)

    def _row_to_decision(self, row: sqlite3.Row) -> PendingDecision:
        return PendingDecision(
            debate_id=row["debate_id"],
            agents=json.loads(row["agents"]),
            consensus_confidence=row["consensus_confidence"],
            consensus_text=row["consensus_text"] or "",
            domain=row["domain"] or "general",
            task=row["task"] or "",
            created_at=row["created_at"],
            verified=bool(row["verified"]),
            verified_at=row["verified_at"],
        )

    def _update_calibration(
        self,
        agents: list[str],
        confidence: float,
        correct: bool,
        domain: str,
        debate_id: str,
    ) -> None:
        """Feed outcome to CalibrationTracker for per-agent Brier scores."""
        try:
            from aragora.agents.calibration import CalibrationTracker

            tracker = CalibrationTracker()
            for agent in agents:
                tracker.record_prediction(
                    agent=agent,
                    confidence=confidence,
                    correct=correct,
                    domain=domain,
                    debate_id=debate_id,
                )
            logger.debug("Updated calibration for %d agents in domain=%s", len(agents), domain)
        except (ImportError, OSError, sqlite3.Error) as e:
            logger.debug("CalibrationTracker update skipped: %s", e)

    def _update_elo(
        self,
        agents: list[str],
        confidence: float,
        correct: bool,
    ) -> None:
        """Feed outcome to ELO system for agent rating adjustments."""
        try:
            from aragora.ranking.elo import EloSystem

            elo = EloSystem()
            correct_float = 1.0 if correct else 0.0
            brier = (confidence - correct_float) ** 2

            for agent in agents:
                rating = elo.get_rating(agent)
                rating.calibration_total += 1
                if correct:
                    rating.calibration_correct += 1
                rating.calibration_brier_sum += brier
                rating.updated_at = datetime.now().isoformat()
                elo._save_rating(rating)

            logger.debug("Updated ELO calibration for %d agents", len(agents))
        except (ImportError, OSError, sqlite3.Error, AttributeError) as e:
            logger.debug("ELO update skipped: %s", e)

    def _update_outcome_tracker(
        self,
        decision: PendingDecision,
        correct: bool,
        signal_detail: str,
    ) -> None:
        """Feed outcome to OutcomeTracker for institutional memory."""
        try:
            from aragora.debate.outcome_tracker import ConsensusOutcome, OutcomeTracker

            tracker = OutcomeTracker()
            outcome = ConsensusOutcome(
                debate_id=decision.debate_id,
                consensus_text=decision.consensus_text,
                consensus_confidence=decision.consensus_confidence,
                implementation_attempted=True,
                implementation_succeeded=correct,
                failure_reason=signal_detail if not correct else None,
                agents_participating=decision.agents,
            )
            tracker.record_outcome(outcome)
            logger.debug("Recorded outcome for debate_id=%s", decision.debate_id)
        except (ImportError, OSError, sqlite3.Error) as e:
            logger.debug("OutcomeTracker update skipped: %s", e)

    def _update_systematic_errors(
        self,
        agents: list[str],
        domain: str,
        confidence: float,
        correct: bool,
    ) -> None:
        """Track per-domain, per-agent error patterns."""
        correct_float = 1.0 if correct else 0.0
        brier = (confidence - correct_float) ** 2
        now = datetime.now().isoformat()

        with self._db.connection() as conn:
            for agent in agents:
                conn.execute(
                    """
                    INSERT INTO systematic_errors (
                        domain, agent, total_verifications, correct_count,
                        brier_sum, avg_confidence, last_updated
                    ) VALUES (?, ?, 1, ?, ?, ?, ?)
                    ON CONFLICT(domain, agent) DO UPDATE SET
                        total_verifications = total_verifications + 1,
                        correct_count = correct_count + excluded.correct_count,
                        brier_sum = brier_sum + excluded.brier_sum,
                        avg_confidence = (
                            avg_confidence * total_verifications + excluded.avg_confidence
                        ) / (total_verifications + 1),
                        last_updated = excluded.last_updated
                    """,
                    (domain, agent, int(correct), brier, confidence, now),
                )
            conn.commit()

    def _compute_calibration_adjustment(self) -> float:
        """Compute system-wide calibration adjustment from recent verifications."""
        with self._db.connection() as conn:
            row = conn.execute("""
                SELECT
                    AVG(p.consensus_confidence) as avg_conf,
                    AVG(CASE WHEN v.outcome_correct = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
                    COUNT(*) as count
                FROM verifications v
                JOIN pending_decisions p ON v.debate_id = p.debate_id
                WHERE p.consensus_confidence >= 0.7
            """).fetchone()

        count = row[2] or 0
        if count < 5:
            return 1.0

        avg_conf = row[0] or 0.0
        success_rate = row[1] or 0.0
        error = avg_conf - success_rate

        # Positive error = overconfident → adjustment < 1.0
        return max(0.5, min(1.5, 1.0 - error))


class AsyncOutcomeVerifier:
    """Async wrapper for OutcomeVerifier."""

    def __init__(self, db_path: Path | None = None):
        self._sync = OutcomeVerifier(db_path)
        self._executor = None

    async def _run(self, func: functools.partial) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, func)

    async def record_decision(self, **kwargs: Any) -> PendingDecision:
        return await self._run(functools.partial(self._sync.record_decision, **kwargs))

    async def verify(self, **kwargs: Any) -> VerificationResult | None:
        return await self._run(functools.partial(self._sync.verify, **kwargs))

    async def get_pending_decisions(self, limit: int = 50) -> list[PendingDecision]:
        return await self._run(functools.partial(self._sync.get_pending_decisions, limit))

    async def get_systematic_errors(self, **kwargs: Any) -> list[dict[str, Any]]:
        return await self._run(functools.partial(self._sync.get_systematic_errors, **kwargs))

    async def get_overall_stats(self) -> dict[str, Any]:
        return await self._run(functools.partial(self._sync.get_overall_stats))

    @property
    def db_path(self) -> Path:
        return self._sync.db_path
