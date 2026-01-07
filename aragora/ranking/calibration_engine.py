"""
Calibration Engine for tournament prediction scoring.

Extracted from EloSystem to separate calibration concerns from competitive ranking.
Handles tournament winner predictions, Brier scores, and calibration leaderboards.

Also includes domain-specific calibration tracking for grounded personas.
"""

import logging
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from aragora.config import (
    DB_TIMEOUT_SECONDS,
    ELO_CALIBRATION_MIN_COUNT,
)

if TYPE_CHECKING:
    from aragora.ranking.elo import EloSystem, AgentRating

logger = logging.getLogger(__name__)

# Use centralized config
CALIBRATION_MIN_COUNT = ELO_CALIBRATION_MIN_COUNT


@dataclass
class CalibrationPrediction:
    """A single calibration prediction record."""

    tournament_id: str
    predictor_agent: str
    predicted_winner: str
    confidence: float
    created_at: Optional[str] = None


@dataclass
class BucketStats:
    """Statistics for a confidence bucket."""

    bucket_key: str
    bucket_start: float
    bucket_end: float
    predictions: int
    correct: int
    accuracy: float
    expected_accuracy: float
    brier_score: float


class CalibrationEngine:
    """
    Tournament prediction calibration scoring engine.

    Manages calibration predictions and scoring independently from ELO ratings.
    Uses Brier score to measure prediction accuracy.

    Usage:
        engine = CalibrationEngine(db_path, elo_system)
        engine.record_winner_prediction(tournament_id, "claude", "gemini", 0.7)
        scores = engine.resolve_tournament(tournament_id, "gemini")
        leaderboard = engine.get_leaderboard()
    """

    def __init__(
        self,
        db_path: Path | str,
        elo_system: Optional["EloSystem"] = None,
    ):
        """
        Initialize the calibration engine.

        Args:
            db_path: Path to database file (same as EloSystem)
            elo_system: Optional EloSystem for updating agent calibration stats
        """
        self.db_path = Path(db_path)
        self.elo_system = elo_system

    def record_winner_prediction(
        self,
        tournament_id: str,
        predictor_agent: str,
        predicted_winner: str,
        confidence: float,
    ) -> None:
        """
        Record an agent's prediction for a tournament winner.

        Args:
            tournament_id: Unique tournament identifier
            predictor_agent: Agent making the prediction
            predicted_winner: Agent predicted to win
            confidence: Confidence level (0.0 to 1.0)
        """
        confidence = min(1.0, max(0.0, confidence))

        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO calibration_predictions
                    (tournament_id, predictor_agent, predicted_winner, confidence)
                VALUES (?, ?, ?, ?)
                """,
                (tournament_id, predictor_agent, predicted_winner, confidence),
            )
            conn.commit()

    def resolve_tournament(
        self,
        tournament_id: str,
        actual_winner: str,
    ) -> dict[str, float]:
        """
        Resolve a tournament and update calibration scores for predictors.

        Uses Brier score: (predicted_probability - actual_outcome)^2
        Lower Brier = better calibration.

        Args:
            tournament_id: Tournament that completed
            actual_winner: Agent who actually won

        Returns:
            Dict of predictor_agent -> brier_score for this prediction
        """
        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT predictor_agent, predicted_winner, confidence
                FROM calibration_predictions
                WHERE tournament_id = ?
                """,
                (tournament_id,),
            )
            predictions = cursor.fetchall()

        if not predictions:
            return {}

        brier_scores: dict[str, float] = {}

        # If we have an ELO system, batch load and update ratings
        if self.elo_system:
            predictor_names = [p[0] for p in predictions]
            ratings = {name: self.elo_system.get_rating(name) for name in predictor_names}

            now = datetime.now().isoformat()
            for predictor, predicted, confidence in predictions:
                # Brier score: (confidence - outcome)^2 where outcome is 1 if correct, 0 if wrong
                correct = 1.0 if predicted == actual_winner else 0.0
                brier = (confidence - correct) ** 2
                brier_scores[predictor] = brier

                # Update the predictor's calibration stats in memory
                rating = ratings[predictor]
                rating.calibration_total += 1
                if predicted == actual_winner:
                    rating.calibration_correct += 1
                rating.calibration_brier_sum += brier
                rating.updated_at = now

            # Batch save all updated ratings in single transaction
            self.elo_system._save_ratings_batch(list(ratings.values()))
        else:
            # Just calculate scores without updating
            for predictor, predicted, confidence in predictions:
                correct = 1.0 if predicted == actual_winner else 0.0
                brier = (confidence - correct) ** 2
                brier_scores[predictor] = brier

        return brier_scores

    def get_leaderboard(self, limit: int = 20) -> list["AgentRating"]:
        """
        Get agents ranked by calibration score.

        Only includes agents with minimum predictions.

        Args:
            limit: Maximum number of agents to return

        Returns:
            List of AgentRating objects sorted by calibration score
        """
        if not self.elo_system:
            return []

        # Delegate to EloSystem for now (uses same table)
        return self.elo_system.get_calibration_leaderboard(limit)

    def get_agent_history(
        self,
        agent_name: str,
        limit: int = 50,
    ) -> list[CalibrationPrediction]:
        """
        Get recent predictions made by an agent.

        Args:
            agent_name: Agent to query
            limit: Maximum predictions to return

        Returns:
            List of CalibrationPrediction objects
        """
        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT tournament_id, predictor_agent, predicted_winner, confidence, created_at
                FROM calibration_predictions
                WHERE predictor_agent = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (agent_name, limit),
            )
            rows = cursor.fetchall()

        return [
            CalibrationPrediction(
                tournament_id=row[0],
                predictor_agent=row[1],
                predicted_winner=row[2],
                confidence=row[3],
                created_at=row[4],
            )
            for row in rows
        ]

    def calculate_brier_score(
        self,
        confidence: float,
        correct: bool,
    ) -> float:
        """
        Calculate Brier score for a single prediction.

        Args:
            confidence: Predicted probability (0.0 to 1.0)
            correct: Whether the prediction was correct

        Returns:
            Brier score (0.0 = perfect, 1.0 = worst)
        """
        outcome = 1.0 if correct else 0.0
        return (confidence - outcome) ** 2


class DomainCalibrationEngine:
    """
    Domain-specific prediction calibration tracking.

    Tracks calibration across different subject domains for grounded personas.
    Provides calibration curves and Expected Calibration Error (ECE).

    Usage:
        engine = DomainCalibrationEngine(db_path, elo_system)
        engine.record_prediction("claude", "security", 0.8, correct=True)
        stats = engine.get_domain_stats("claude")
        ece = engine.get_expected_calibration_error("claude")
    """

    def __init__(
        self,
        db_path: Path | str,
        elo_system: Optional["EloSystem"] = None,
    ):
        """
        Initialize the domain calibration engine.

        Args:
            db_path: Path to database file
            elo_system: Optional EloSystem for updating overall calibration stats
        """
        self.db_path = Path(db_path)
        self.elo_system = elo_system

    @staticmethod
    def get_bucket_key(confidence: float) -> str:
        """
        Convert confidence to bucket key.

        Args:
            confidence: Value between 0.0 and 1.0

        Returns:
            Bucket key string (e.g., "0.7-0.8")
        """
        bucket_start = math.floor(confidence * 10) / 10
        bucket_end = min(1.0, bucket_start + 0.1)
        return f"{bucket_start:.1f}-{bucket_end:.1f}"

    def record_prediction(
        self,
        agent_name: str,
        domain: str,
        confidence: float,
        correct: bool,
    ) -> None:
        """
        Record a domain-specific prediction for calibration tracking.

        Args:
            agent_name: Agent making the prediction
            domain: Domain/topic area (e.g., "security", "performance")
            confidence: Confidence level (0.0 to 1.0)
            correct: Whether the prediction was correct
        """
        confidence = min(1.0, max(0.0, confidence))
        brier = (confidence - (1.0 if correct else 0.0)) ** 2
        bucket_key = self.get_bucket_key(confidence)
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
            cursor = conn.cursor()

            # Update domain calibration
            cursor.execute(
                """
                INSERT INTO domain_calibration (agent_name, domain, total_predictions, total_correct, brier_sum, updated_at)
                VALUES (?, ?, 1, ?, ?, ?)
                ON CONFLICT(agent_name, domain) DO UPDATE SET
                    total_predictions = total_predictions + 1,
                    total_correct = total_correct + ?,
                    brier_sum = brier_sum + ?,
                    updated_at = ?
                """,
                (
                    agent_name, domain, 1 if correct else 0, brier, now,
                    1 if correct else 0, brier, now,
                ),
            )

            # Update calibration bucket
            cursor.execute(
                """
                INSERT INTO calibration_buckets (agent_name, domain, bucket_key, predictions, correct, brier_sum)
                VALUES (?, ?, ?, 1, ?, ?)
                ON CONFLICT(agent_name, domain, bucket_key) DO UPDATE SET
                    predictions = predictions + 1,
                    correct = correct + ?,
                    brier_sum = brier_sum + ?
                """,
                (agent_name, domain, bucket_key, 1 if correct else 0, brier, 1 if correct else 0, brier),
            )

            conn.commit()

        # Update overall calibration stats via EloSystem
        if self.elo_system:
            rating = self.elo_system.get_rating(agent_name)
            rating.calibration_total += 1
            if correct:
                rating.calibration_correct += 1
            rating.calibration_brier_sum += brier
            rating.updated_at = now
            self.elo_system._save_rating(rating)

    def get_domain_stats(
        self,
        agent_name: str,
        domain: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Get calibration statistics for an agent.

        Args:
            agent_name: Agent to query
            domain: Optional domain filter

        Returns:
            Dict with total, correct, accuracy, brier_score, and per-domain breakdown
        """
        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
            cursor = conn.cursor()
            if domain:
                cursor.execute(
                    """SELECT domain, total_predictions, total_correct, brier_sum
                       FROM domain_calibration WHERE agent_name = ? AND domain = ?""",
                    (agent_name, domain),
                )
            else:
                cursor.execute(
                    """SELECT domain, total_predictions, total_correct, brier_sum
                       FROM domain_calibration WHERE agent_name = ?
                       ORDER BY total_predictions DESC""",
                    (agent_name,),
                )
            rows = cursor.fetchall()

        if not rows:
            return {"total": 0, "correct": 0, "accuracy": 0.0, "brier_score": 1.0, "domains": {}}

        domains: dict[str, dict] = {}
        total_predictions = 0
        total_correct = 0
        total_brier = 0.0

        for row in rows:
            domain_name, predictions, correct, brier = row
            domains[domain_name] = {
                "predictions": predictions,
                "correct": correct,
                "accuracy": correct / predictions if predictions > 0 else 0.0,
                "brier_score": brier / predictions if predictions > 0 else 1.0,
            }
            total_predictions += predictions
            total_correct += correct
            total_brier += brier

        return {
            "total": total_predictions,
            "correct": total_correct,
            "accuracy": total_correct / total_predictions if total_predictions > 0 else 0.0,
            "brier_score": total_brier / total_predictions if total_predictions > 0 else 1.0,
            "domains": domains,
        }

    def get_calibration_curve(
        self,
        agent_name: str,
        domain: Optional[str] = None,
    ) -> list[BucketStats]:
        """
        Get calibration broken down by confidence bucket.

        Used for plotting calibration curves.

        Args:
            agent_name: Agent to query
            domain: Optional domain filter

        Returns:
            List of BucketStats for each confidence bucket
        """
        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
            cursor = conn.cursor()
            if domain:
                cursor.execute(
                    """SELECT bucket_key, SUM(predictions), SUM(correct), SUM(brier_sum)
                       FROM calibration_buckets WHERE agent_name = ? AND domain = ?
                       GROUP BY bucket_key ORDER BY bucket_key""",
                    (agent_name, domain),
                )
            else:
                cursor.execute(
                    """SELECT bucket_key, SUM(predictions), SUM(correct), SUM(brier_sum)
                       FROM calibration_buckets WHERE agent_name = ?
                       GROUP BY bucket_key ORDER BY bucket_key""",
                    (agent_name,),
                )
            rows = cursor.fetchall()

        buckets = []
        for row in rows:
            bucket_key, predictions, correct, brier = row
            parts = bucket_key.split("-")
            if len(parts) < 2:
                logger.warning(f"Malformed bucket key: {bucket_key}, skipping")
                continue
            try:
                bucket_start = float(parts[0])
                bucket_end = float(parts[1])
            except ValueError:
                logger.warning(f"Invalid bucket values in {bucket_key}, skipping")
                continue

            expected = (bucket_start + bucket_end) / 2

            buckets.append(BucketStats(
                bucket_key=bucket_key,
                bucket_start=bucket_start,
                bucket_end=bucket_end,
                predictions=predictions,
                correct=correct,
                accuracy=correct / predictions if predictions > 0 else 0.0,
                expected_accuracy=expected,
                brier_score=brier / predictions if predictions > 0 else 1.0,
            ))

        return buckets

    def get_expected_calibration_error(self, agent_name: str) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        Lower ECE means better calibration (0 = perfect).

        Args:
            agent_name: Agent to query

        Returns:
            ECE value (0.0 to 1.0)
        """
        buckets = self.get_calibration_curve(agent_name)
        if not buckets:
            return 1.0

        total_predictions = sum(b.predictions for b in buckets)
        if total_predictions == 0:
            return 1.0

        ece = 0.0
        for bucket in buckets:
            weight = bucket.predictions / total_predictions
            calibration_error = abs(bucket.accuracy - bucket.expected_accuracy)
            ece += weight * calibration_error

        return ece

    def get_best_domains(
        self,
        agent_name: str,
        limit: int = 5,
        min_predictions: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Get domains where agent is best calibrated.

        Args:
            agent_name: Agent to query
            limit: Maximum domains to return
            min_predictions: Minimum predictions required per domain

        Returns:
            List of (domain, score) tuples sorted by score descending
        """
        stats = self.get_domain_stats(agent_name)
        domains = stats.get("domains", {})

        scored = []
        for domain, domain_stats in domains.items():
            if domain_stats["predictions"] < min_predictions:
                continue
            # Confidence increases with more predictions
            confidence = min(1.0, 0.5 + 0.5 * (domain_stats["predictions"] - min_predictions) / 20)
            score = (1 - domain_stats["brier_score"]) * confidence
            scored.append((domain, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]
