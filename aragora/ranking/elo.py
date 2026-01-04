"""
ELO/Reputation System for agent skill tracking.

Inspired by ChatArena's competitive environments, this module provides:
- ELO ratings for agents
- Domain-specific skill ratings
- Match history and statistics
- Leaderboards
"""

import json
import sqlite3
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# Default ELO rating for new agents
DEFAULT_ELO = 1500
K_FACTOR = 32  # How quickly ratings change

# Calibration scoring constants
CALIBRATION_MIN_COUNT = 10  # Minimum predictions for meaningful score


@dataclass
class AgentRating:
    """An agent's rating and statistics."""

    agent_name: str
    elo: float = DEFAULT_ELO
    domain_elos: dict[str, float] = field(default_factory=dict)
    wins: int = 0
    losses: int = 0
    draws: int = 0
    debates_count: int = 0
    critiques_accepted: int = 0
    critiques_total: int = 0
    # Calibration scoring fields
    calibration_correct: int = 0
    calibration_total: int = 0
    calibration_brier_sum: float = 0.0
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        total = self.wins + self.losses + self.draws
        return self.wins / total if total > 0 else 0.0

    @property
    def critique_acceptance_rate(self) -> float:
        """Calculate critique acceptance rate."""
        return self.critiques_accepted / self.critiques_total if self.critiques_total > 0 else 0.0

    @property
    def games_played(self) -> int:
        """Total games played."""
        return self.wins + self.losses + self.draws

    @property
    def calibration_accuracy(self) -> float:
        """Fraction of correct winner predictions."""
        if self.calibration_total == 0:
            return 0.0
        return self.calibration_correct / self.calibration_total

    @property
    def calibration_brier_score(self) -> float:
        """Average Brier score (lower is better, 0 = perfect)."""
        if self.calibration_total == 0:
            return 1.0
        return self.calibration_brier_sum / self.calibration_total

    @property
    def calibration_score(self) -> float:
        """
        Combined calibration score (higher is better).

        Uses (1 - Brier) weighted by confidence from sample size.
        Requires minimum predictions for meaningful score.
        """
        if self.calibration_total < CALIBRATION_MIN_COUNT:
            return 0.0
        # Confidence scales from 0.5 at min_count to 1.0 at 50+ predictions
        confidence = min(1.0, 0.5 + 0.5 * (self.calibration_total - CALIBRATION_MIN_COUNT) / 40)
        return (1 - self.calibration_brier_score) * confidence


@dataclass
class MatchResult:
    """Result of a debate match between agents."""

    debate_id: str
    winner: Optional[str]  # None for draw
    participants: list[str]
    domain: Optional[str]
    scores: dict[str, float]  # agent -> score
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class EloSystem:
    """
    ELO-based ranking system for agents.

    Tracks agent skill ratings, match history, and provides leaderboards.
    """

    def __init__(self, db_path: str = "aragora_elo.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Agent ratings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ratings (
                agent_name TEXT PRIMARY KEY,
                elo REAL DEFAULT 1500,
                domain_elos TEXT,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                draws INTEGER DEFAULT 0,
                debates_count INTEGER DEFAULT 0,
                critiques_accepted INTEGER DEFAULT 0,
                critiques_total INTEGER DEFAULT 0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Match history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                debate_id TEXT UNIQUE,
                winner TEXT,
                participants TEXT,
                domain TEXT,
                scores TEXT,
                elo_changes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ELO history for tracking progression
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS elo_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                elo REAL NOT NULL,
                debate_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Calibration predictions table (for tracking pre-tournament predictions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibration_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tournament_id TEXT NOT NULL,
                predictor_agent TEXT NOT NULL,
                predicted_winner TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(tournament_id, predictor_agent)
            )
        """)

        # Safe schema migration: add calibration columns if missing
        cursor.execute("PRAGMA table_info(ratings)")
        columns = {row[1] for row in cursor.fetchall()}
        if "calibration_correct" not in columns:
            cursor.execute("ALTER TABLE ratings ADD COLUMN calibration_correct INTEGER DEFAULT 0")
        if "calibration_total" not in columns:
            cursor.execute("ALTER TABLE ratings ADD COLUMN calibration_total INTEGER DEFAULT 0")
        if "calibration_brier_sum" not in columns:
            cursor.execute("ALTER TABLE ratings ADD COLUMN calibration_brier_sum REAL DEFAULT 0.0")

        conn.commit()
        conn.close()

    def get_rating(self, agent_name: str) -> AgentRating:
        """Get or create rating for an agent."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT agent_name, elo, domain_elos, wins, losses, draws,
                   debates_count, critiques_accepted, critiques_total,
                   calibration_correct, calibration_total, calibration_brier_sum,
                   updated_at
            FROM ratings WHERE agent_name = ?
            """,
            (agent_name,),
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return AgentRating(agent_name=agent_name)

        return AgentRating(
            agent_name=row[0],
            elo=row[1],
            domain_elos=json.loads(row[2]) if row[2] else {},
            wins=row[3],
            losses=row[4],
            draws=row[5],
            debates_count=row[6],
            critiques_accepted=row[7],
            critiques_total=row[8],
            calibration_correct=row[9] or 0,
            calibration_total=row[10] or 0,
            calibration_brier_sum=row[11] or 0.0,
            updated_at=row[12],
        )

    def _save_rating(self, rating: AgentRating):
        """Save rating to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO ratings (agent_name, elo, domain_elos, wins, losses, draws,
                                debates_count, critiques_accepted, critiques_total,
                                calibration_correct, calibration_total, calibration_brier_sum,
                                updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_name) DO UPDATE SET
                elo = excluded.elo,
                domain_elos = excluded.domain_elos,
                wins = excluded.wins,
                losses = excluded.losses,
                draws = excluded.draws,
                debates_count = excluded.debates_count,
                critiques_accepted = excluded.critiques_accepted,
                critiques_total = excluded.critiques_total,
                calibration_correct = excluded.calibration_correct,
                calibration_total = excluded.calibration_total,
                calibration_brier_sum = excluded.calibration_brier_sum,
                updated_at = excluded.updated_at
            """,
            (
                rating.agent_name,
                rating.elo,
                json.dumps(rating.domain_elos),
                rating.wins,
                rating.losses,
                rating.draws,
                rating.debates_count,
                rating.critiques_accepted,
                rating.critiques_total,
                rating.calibration_correct,
                rating.calibration_total,
                rating.calibration_brier_sum,
                rating.updated_at,
            ),
        )

        conn.commit()
        conn.close()

    def _expected_score(self, elo_a: float, elo_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    def _calculate_new_elo(
        self,
        current_elo: float,
        expected: float,
        actual: float,
        k: float = K_FACTOR,
    ) -> float:
        """Calculate new ELO rating."""
        return current_elo + k * (actual - expected)

    def record_match(
        self,
        debate_id: str,
        participants: list[str],
        scores: dict[str, float],
        domain: str = None,
    ) -> dict[str, float]:
        """
        Record a match result and update ELO ratings.

        Args:
            debate_id: Unique debate identifier
            participants: List of agent names
            scores: Dict of agent -> score (higher is better)
            domain: Optional domain for domain-specific ELO

        Returns:
            Dict of agent -> ELO change
        """
        if len(participants) < 2:
            return {}

        # Determine winner (highest score)
        sorted_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        winner = sorted_agents[0][0] if sorted_agents[0][1] > sorted_agents[1][1] else None

        # Get current ratings
        ratings = {name: self.get_rating(name) for name in participants}
        elo_changes = {}

        # Calculate pairwise ELO updates
        for i, agent_a in enumerate(participants):
            for agent_b in participants[i + 1:]:
                rating_a = ratings[agent_a]
                rating_b = ratings[agent_b]

                # Expected scores
                expected_a = self._expected_score(rating_a.elo, rating_b.elo)
                expected_b = 1 - expected_a

                # Actual scores (normalized to 0-1)
                score_a = scores.get(agent_a, 0)
                score_b = scores.get(agent_b, 0)
                total = score_a + score_b
                if total > 0:
                    actual_a = score_a / total
                    actual_b = score_b / total
                else:
                    actual_a = actual_b = 0.5

                # Update ELOs
                change_a = K_FACTOR * (actual_a - expected_a)
                change_b = K_FACTOR * (actual_b - expected_b)

                elo_changes[agent_a] = elo_changes.get(agent_a, 0) + change_a
                elo_changes[agent_b] = elo_changes.get(agent_b, 0) + change_b

        # Apply changes and update stats
        for agent_name, change in elo_changes.items():
            rating = ratings[agent_name]
            rating.elo += change
            rating.debates_count += 1
            rating.updated_at = datetime.now().isoformat()

            if winner == agent_name:
                rating.wins += 1
            elif winner is None:
                rating.draws += 1
            else:
                rating.losses += 1

            # Update domain ELO if specified
            if domain:
                current_domain_elo = rating.domain_elos.get(domain, DEFAULT_ELO)
                rating.domain_elos[domain] = current_domain_elo + change

            self._save_rating(rating)
            self._record_elo_history(agent_name, rating.elo, debate_id)

        # Save match
        self._save_match(debate_id, winner, participants, domain, scores, elo_changes)

        return elo_changes

    def _save_match(
        self,
        debate_id: str,
        winner: Optional[str],
        participants: list[str],
        domain: Optional[str],
        scores: dict[str, float],
        elo_changes: dict[str, float],
    ):
        """Save match to history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO matches (debate_id, winner, participants, domain, scores, elo_changes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                debate_id,
                winner,
                json.dumps(participants),
                domain,
                json.dumps(scores),
                json.dumps(elo_changes),
            ),
        )

        conn.commit()
        conn.close()

    def _record_elo_history(self, agent_name: str, elo: float, debate_id: str = None):
        """Record ELO at a point in time."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO elo_history (agent_name, elo, debate_id) VALUES (?, ?, ?)",
            (agent_name, elo, debate_id),
        )

        conn.commit()
        conn.close()

    def record_critique(self, agent_name: str, accepted: bool):
        """Record a critique and whether it was accepted."""
        rating = self.get_rating(agent_name)
        rating.critiques_total += 1
        if accepted:
            rating.critiques_accepted += 1
        rating.updated_at = datetime.now().isoformat()
        self._save_rating(rating)

    def get_leaderboard(self, limit: int = 20, domain: str = None) -> list[AgentRating]:
        """Get top agents by ELO."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT agent_name, elo, domain_elos, wins, losses, draws,
                   debates_count, critiques_accepted, critiques_total, updated_at
            FROM ratings
            ORDER BY elo DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()

        ratings = [
            AgentRating(
                agent_name=row[0],
                elo=row[1],
                domain_elos=json.loads(row[2]) if row[2] else {},
                wins=row[3],
                losses=row[4],
                draws=row[5],
                debates_count=row[6],
                critiques_accepted=row[7],
                critiques_total=row[8],
                updated_at=row[9],
            )
            for row in rows
        ]

        # Re-sort by domain ELO if specified
        if domain:
            ratings.sort(
                key=lambda r: r.domain_elos.get(domain, DEFAULT_ELO),
                reverse=True,
            )

        return ratings

    def get_elo_history(self, agent_name: str, limit: int = 50) -> list[tuple[str, float]]:
        """Get ELO history for an agent."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT created_at, elo FROM elo_history
            WHERE agent_name = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (agent_name, limit),
        )
        rows = cursor.fetchall()
        conn.close()

        return [(row[0], row[1]) for row in rows]

    def get_recent_matches(self, limit: int = 10) -> list[dict]:
        """Get recent match results with ELO changes.

        Returns list of dicts with: debate_id, winner, participants, domain,
        elo_changes, created_at
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT debate_id, winner, participants, domain, elo_changes, created_at
            FROM matches
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()

        matches = []
        for row in rows:
            elo_changes = json.loads(row[4]) if row[4] else {}
            participants = json.loads(row[2]) if row[2] else []
            matches.append({
                "debate_id": row[0],
                "winner": row[1],
                "participants": participants,
                "domain": row[3],
                "elo_changes": elo_changes,
                "created_at": row[5],
            })
        return matches

    def get_head_to_head(self, agent_a: str, agent_b: str) -> dict:
        """Get head-to-head statistics between two agents."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT winner, scores FROM matches
            WHERE participants LIKE ? AND participants LIKE ?
            """,
            (f"%{agent_a}%", f"%{agent_b}%"),
        )
        rows = cursor.fetchall()
        conn.close()

        a_wins = 0
        b_wins = 0
        draws = 0

        for winner, _ in rows:
            if winner == agent_a:
                a_wins += 1
            elif winner == agent_b:
                b_wins += 1
            else:
                draws += 1

        return {
            "matches": len(rows),
            f"{agent_a}_wins": a_wins,
            f"{agent_b}_wins": b_wins,
            "draws": draws,
        }

    def get_stats(self) -> dict:
        """Get overall system statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*), AVG(elo), MAX(elo), MIN(elo) FROM ratings")
        ratings_row = cursor.fetchone()

        cursor.execute("SELECT COUNT(*) FROM matches")
        matches_row = cursor.fetchone()

        conn.close()

        return {
            "total_agents": ratings_row[0] or 0,
            "avg_elo": ratings_row[1] or DEFAULT_ELO,
            "max_elo": ratings_row[2] or DEFAULT_ELO,
            "min_elo": ratings_row[3] or DEFAULT_ELO,
            "total_matches": matches_row[0] or 0,
        }

    # =========================================================================
    # Tournament Winner Calibration Scoring
    # =========================================================================

    def record_winner_prediction(
        self,
        tournament_id: str,
        predictor_agent: str,
        predicted_winner: str,
        confidence: float,
    ):
        """
        Record an agent's prediction for a tournament winner.

        Args:
            tournament_id: Unique tournament identifier
            predictor_agent: Agent making the prediction
            predicted_winner: Agent predicted to win
            confidence: Confidence level (0.0 to 1.0)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO calibration_predictions
                (tournament_id, predictor_agent, predicted_winner, confidence)
            VALUES (?, ?, ?, ?)
            """,
            (tournament_id, predictor_agent, predicted_winner, min(1.0, max(0.0, confidence))),
        )

        conn.commit()
        conn.close()

    def resolve_tournament_calibration(
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
        conn = sqlite3.connect(self.db_path)
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
        conn.close()

        brier_scores = {}

        for predictor, predicted, confidence in predictions:
            # Brier score: (confidence - outcome)^2 where outcome is 1 if correct, 0 if wrong
            correct = 1.0 if predicted == actual_winner else 0.0
            brier = (confidence - correct) ** 2
            brier_scores[predictor] = brier

            # Update the predictor's calibration stats
            rating = self.get_rating(predictor)
            rating.calibration_total += 1
            if predicted == actual_winner:
                rating.calibration_correct += 1
            rating.calibration_brier_sum += brier
            rating.updated_at = datetime.now().isoformat()
            self._save_rating(rating)

        return brier_scores

    def get_calibration_leaderboard(self, limit: int = 20) -> list[AgentRating]:
        """
        Get agents ranked by calibration score.

        Only includes agents with minimum predictions.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT agent_name, elo, domain_elos, wins, losses, draws,
                   debates_count, critiques_accepted, critiques_total,
                   calibration_correct, calibration_total, calibration_brier_sum,
                   updated_at
            FROM ratings
            WHERE calibration_total >= ?
            ORDER BY (1.0 - calibration_brier_sum / calibration_total) DESC
            LIMIT ?
            """,
            (CALIBRATION_MIN_COUNT, limit),
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            AgentRating(
                agent_name=row[0],
                elo=row[1],
                domain_elos=json.loads(row[2]) if row[2] else {},
                wins=row[3],
                losses=row[4],
                draws=row[5],
                debates_count=row[6],
                critiques_accepted=row[7],
                critiques_total=row[8],
                calibration_correct=row[9] or 0,
                calibration_total=row[10] or 0,
                calibration_brier_sum=row[11] or 0.0,
                updated_at=row[12],
            )
            for row in rows
        ]

    def get_agent_calibration_history(
        self, agent_name: str, limit: int = 50
    ) -> list[dict]:
        """Get recent predictions made by an agent."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT tournament_id, predicted_winner, confidence, created_at
            FROM calibration_predictions
            WHERE predictor_agent = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (agent_name, limit),
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "tournament_id": row[0],
                "predicted_winner": row[1],
                "confidence": row[2],
                "created_at": row[3],
            }
            for row in rows
        ]
