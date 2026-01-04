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


def _escape_like_pattern(value: str) -> str:
    """Escape special characters in SQL LIKE patterns to prevent injection."""
    return value.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')


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
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
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

            # Domain-specific calibration tracking (for grounded personas)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS domain_calibration (
                    agent_name TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    total_predictions INTEGER DEFAULT 0,
                    total_correct INTEGER DEFAULT 0,
                    brier_sum REAL DEFAULT 0.0,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (agent_name, domain)
                )
            """)

            # Calibration by confidence bucket (for calibration curves)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS calibration_buckets (
                    agent_name TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    bucket_key TEXT NOT NULL,
                    predictions INTEGER DEFAULT 0,
                    correct INTEGER DEFAULT 0,
                    brier_sum REAL DEFAULT 0.0,
                    PRIMARY KEY (agent_name, domain, bucket_key)
                )
            """)

            # Agent relationships tracking (for grounded personas)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_relationships (
                    agent_a TEXT NOT NULL,
                    agent_b TEXT NOT NULL,
                    debate_count INTEGER DEFAULT 0,
                    agreement_count INTEGER DEFAULT 0,
                    critique_count_a_to_b INTEGER DEFAULT 0,
                    critique_count_b_to_a INTEGER DEFAULT 0,
                    critique_accepted_a_to_b INTEGER DEFAULT 0,
                    critique_accepted_b_to_a INTEGER DEFAULT 0,
                    position_changes_a_after_b INTEGER DEFAULT 0,
                    position_changes_b_after_a INTEGER DEFAULT 0,
                    a_wins_over_b INTEGER DEFAULT 0,
                    b_wins_over_a INTEGER DEFAULT 0,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (agent_a, agent_b),
                    CHECK (agent_a < agent_b)
                )
            """)


    def get_rating(self, agent_name: str) -> AgentRating:
        """Get or create rating for an agent."""
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
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

        # Write JSON snapshot for fast reads (avoids SQLite locking)
        self._write_snapshot()

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
        with sqlite3.connect(self.db_path) as conn:
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

    def _record_elo_history(self, agent_name: str, elo: float, debate_id: str = None):
        """Record ELO at a point in time."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO elo_history (agent_name, elo, debate_id) VALUES (?, ?, ?)",
                (agent_name, elo, debate_id),
            )
            conn.commit()

    def _write_snapshot(self) -> None:
        """Write JSON snapshot for fast reads.

        Creates an atomic JSON file with current leaderboard and recent matches.
        This avoids SQLite locking issues when multiple readers access data.
        """
        snapshot_path = self.db_path.parent / "elo_snapshot.json"

        # Gather current state
        leaderboard = self.get_leaderboard(limit=100)
        data = {
            "leaderboard": [
                {
                    "agent_name": r.agent_name,
                    "elo": r.elo,
                    "wins": r.wins,
                    "losses": r.losses,
                    "draws": r.draws,
                    "games_played": r.games_played,
                    "win_rate": r.win_rate,
                }
                for r in leaderboard
            ],
            "recent_matches": self.get_recent_matches(limit=50),
            "updated_at": datetime.now().isoformat(),
        }

        # Atomic write: write to temp file then rename
        temp_path = snapshot_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(data, f)
            temp_path.rename(snapshot_path)
        except Exception:
            # Snapshot is optional, don't fail on write errors
            if temp_path.exists():
                temp_path.unlink()

    def get_cached_leaderboard(self, limit: int = 20) -> list[dict]:
        """Get leaderboard from cache if available.

        Falls back to database query if cache is missing or stale.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of dicts with agent_name, elo, wins, losses, draws, games_played, win_rate
        """
        snapshot_path = self.db_path.parent / "elo_snapshot.json"
        if snapshot_path.exists():
            try:
                with open(snapshot_path) as f:
                    data = json.load(f)
                return data.get("leaderboard", [])[:limit]
            except Exception:
                pass

        # Fall back to database
        leaderboard = self.get_leaderboard(limit)
        return [
            {
                "agent_name": r.agent_name,
                "elo": r.elo,
                "wins": r.wins,
                "losses": r.losses,
                "draws": r.draws,
                "games_played": r.games_played,
                "win_rate": r.win_rate,
            }
            for r in leaderboard
        ]

    def get_cached_recent_matches(self, limit: int = 10) -> list[dict]:
        """Get recent matches from cache if available.

        Falls back to database query if cache is missing.

        Args:
            limit: Maximum number of matches to return

        Returns:
            List of match dicts
        """
        snapshot_path = self.db_path.parent / "elo_snapshot.json"
        if snapshot_path.exists():
            try:
                with open(snapshot_path) as f:
                    data = json.load(f)
                return data.get("recent_matches", [])[:limit]
            except Exception:
                pass

        # Fall back to database
        return self.get_recent_matches(limit)

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
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:

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
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:

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

        return [(row[0], row[1]) for row in rows]

    def get_recent_matches(self, limit: int = 10) -> list[dict]:
        """Get recent match results with ELO changes.

        Returns list of dicts with: debate_id, winner, participants, domain,
        elo_changes, created_at
        """
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:

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
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()

            # Escape LIKE special characters to prevent SQL injection
            escaped_a = _escape_like_pattern(agent_a)
            escaped_b = _escape_like_pattern(agent_b)

            cursor.execute(
                """
                SELECT winner, scores FROM matches
                WHERE participants LIKE ? ESCAPE '\\' AND participants LIKE ? ESCAPE '\\'
                """,
                (f"%{escaped_a}%", f"%{escaped_b}%"),
            )
            rows = cursor.fetchall()

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
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:

            cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*), AVG(elo), MAX(elo), MIN(elo) FROM ratings")
        ratings_row = cursor.fetchone()

        cursor.execute("SELECT COUNT(*) FROM matches")
        matches_row = cursor.fetchone()


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
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:

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
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:

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
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:

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
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:

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

        return [
            {
                "tournament_id": row[0],
                "predicted_winner": row[1],
                "confidence": row[2],
                "created_at": row[3],
            }
            for row in rows
        ]

    # =========================================================================
    # Domain-Specific Calibration Tracking (Grounded Personas)
    # =========================================================================

    def _get_bucket_key(self, confidence: float) -> str:
        """Convert confidence to bucket key (0.0-0.1 -> '0.0-0.1')."""
        bucket_start = math.floor(confidence * 10) / 10
        bucket_end = min(1.0, bucket_start + 0.1)
        return f"{bucket_start:.1f}-{bucket_end:.1f}"

    def record_domain_prediction(
        self,
        agent_name: str,
        domain: str,
        confidence: float,
        correct: bool,
    ):
        """
        Record a domain-specific prediction for calibration tracking.

        Args:
            agent_name: Agent making the prediction
            domain: Domain/topic area (e.g., "ethics", "code_review", "economics")
            confidence: Confidence level (0.0 to 1.0)
            correct: Whether the prediction was correct
        """
        confidence = min(1.0, max(0.0, confidence))
        brier = (confidence - (1.0 if correct else 0.0)) ** 2
        bucket_key = self._get_bucket_key(confidence)

        with sqlite3.connect(self.db_path, timeout=30.0) as conn:


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
                agent_name, domain, 1 if correct else 0, brier, datetime.now().isoformat(),
                1 if correct else 0, brier, datetime.now().isoformat(),
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

        # Also update overall calibration stats
        rating = self.get_rating(agent_name)
        rating.calibration_total += 1
        if correct:
            rating.calibration_correct += 1
        rating.calibration_brier_sum += brier
        rating.updated_at = datetime.now().isoformat()

        conn.commit()
        self._save_rating(rating)

    def get_domain_calibration(self, agent_name: str, domain: Optional[str] = None) -> dict:
        """Get calibration statistics for an agent, optionally filtered by domain."""
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:

            cursor = conn.cursor()

        if domain:
            cursor.execute(
                "SELECT domain, total_predictions, total_correct, brier_sum FROM domain_calibration WHERE agent_name = ? AND domain = ?",
                (agent_name, domain),
            )
        else:
            cursor.execute(
                "SELECT domain, total_predictions, total_correct, brier_sum FROM domain_calibration WHERE agent_name = ? ORDER BY total_predictions DESC",
                (agent_name,),
            )

        rows = cursor.fetchall()

        if not rows:
            return {"total": 0, "correct": 0, "accuracy": 0.0, "brier_score": 1.0, "domains": {}}

        domains = {}
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

    def get_calibration_by_bucket(self, agent_name: str, domain: Optional[str] = None) -> list[dict]:
        """Get calibration broken down by confidence bucket for calibration curves."""
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:

            cursor = conn.cursor()

        if domain:
            cursor.execute(
                "SELECT bucket_key, SUM(predictions), SUM(correct), SUM(brier_sum) FROM calibration_buckets WHERE agent_name = ? AND domain = ? GROUP BY bucket_key ORDER BY bucket_key",
                (agent_name, domain),
            )
        else:
            cursor.execute(
                "SELECT bucket_key, SUM(predictions), SUM(correct), SUM(brier_sum) FROM calibration_buckets WHERE agent_name = ? GROUP BY bucket_key ORDER BY bucket_key",
                (agent_name,),
            )

        rows = cursor.fetchall()

        buckets = []
        for row in rows:
            bucket_key, predictions, correct, brier = row
            parts = bucket_key.split("-")
            bucket_start = float(parts[0])
            bucket_end = float(parts[1])
            expected = (bucket_start + bucket_end) / 2

            buckets.append({
                "bucket_key": bucket_key,
                "bucket_start": bucket_start,
                "bucket_end": bucket_end,
                "predictions": predictions,
                "correct": correct,
                "accuracy": correct / predictions if predictions > 0 else 0.0,
                "expected_accuracy": expected,
                "brier_score": brier / predictions if predictions > 0 else 1.0,
            })
        return buckets

    def get_expected_calibration_error(self, agent_name: str) -> float:
        """Calculate Expected Calibration Error (ECE) - lower is better (0 = perfect)."""
        buckets = self.get_calibration_by_bucket(agent_name)
        if not buckets:
            return 1.0
        total_predictions = sum(b["predictions"] for b in buckets)
        if total_predictions == 0:
            return 1.0
        ece = 0.0
        for bucket in buckets:
            weight = bucket["predictions"] / total_predictions
            calibration_error = abs(bucket["accuracy"] - bucket["expected_accuracy"])
            ece += weight * calibration_error
        return ece

    def get_best_domains(self, agent_name: str, limit: int = 5) -> list[tuple[str, float]]:
        """Get domains where agent is best calibrated."""
        calibration = self.get_domain_calibration(agent_name)
        domains = calibration.get("domains", {})
        scored = []
        for domain, stats in domains.items():
            if stats["predictions"] < 5:
                continue
            confidence = min(1.0, 0.5 + 0.5 * (stats["predictions"] - 5) / 20)
            score = (1 - stats["brier_score"]) * confidence
            scored.append((domain, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    # =========================================================================
    # Agent Relationship Tracking (Grounded Personas)
    # =========================================================================

    def update_relationship(
        self,
        agent_a: str,
        agent_b: str,
        debate_increment: int = 0,
        agreement_increment: int = 0,
        critique_a_to_b: int = 0,
        critique_b_to_a: int = 0,
        critique_accepted_a_to_b: int = 0,
        critique_accepted_b_to_a: int = 0,
        position_change_a_after_b: int = 0,
        position_change_b_after_a: int = 0,
        a_win: int = 0,
        b_win: int = 0,
    ):
        """Update relationship stats between two agents (maintains canonical a < b ordering)."""
        if agent_a > agent_b:
            agent_a, agent_b = agent_b, agent_a
            critique_a_to_b, critique_b_to_a = critique_b_to_a, critique_a_to_b
            critique_accepted_a_to_b, critique_accepted_b_to_a = critique_accepted_b_to_a, critique_accepted_a_to_b
            position_change_a_after_b, position_change_b_after_a = position_change_b_after_a, position_change_a_after_b
            a_win, b_win = b_win, a_win

        with sqlite3.connect(self.db_path, timeout=30.0) as conn:


            cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO agent_relationships (agent_a, agent_b, debate_count, agreement_count,
                critique_count_a_to_b, critique_count_b_to_a, critique_accepted_a_to_b, critique_accepted_b_to_a,
                position_changes_a_after_b, position_changes_b_after_a, a_wins_over_b, b_wins_over_a, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_a, agent_b) DO UPDATE SET
                debate_count = debate_count + ?, agreement_count = agreement_count + ?,
                critique_count_a_to_b = critique_count_a_to_b + ?, critique_count_b_to_a = critique_count_b_to_a + ?,
                critique_accepted_a_to_b = critique_accepted_a_to_b + ?, critique_accepted_b_to_a = critique_accepted_b_to_a + ?,
                position_changes_a_after_b = position_changes_a_after_b + ?, position_changes_b_after_a = position_changes_b_after_a + ?,
                a_wins_over_b = a_wins_over_b + ?, b_wins_over_a = b_wins_over_a + ?, updated_at = ?
            """,
            (agent_a, agent_b, debate_increment, agreement_increment, critique_a_to_b, critique_b_to_a,
             critique_accepted_a_to_b, critique_accepted_b_to_a, position_change_a_after_b, position_change_b_after_a,
             a_win, b_win, datetime.now().isoformat(),
             debate_increment, agreement_increment, critique_a_to_b, critique_b_to_a,
             critique_accepted_a_to_b, critique_accepted_b_to_a, position_change_a_after_b, position_change_b_after_a,
             a_win, b_win, datetime.now().isoformat()),
        )
        conn.commit()

    def get_relationship_raw(self, agent_a: str, agent_b: str) -> Optional[dict]:
        """Get raw relationship data between two agents."""
        if agent_a > agent_b:
            agent_a, agent_b = agent_b, agent_a
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:

            cursor = conn.cursor()
        cursor.execute(
            "SELECT debate_count, agreement_count, critique_count_a_to_b, critique_count_b_to_a, critique_accepted_a_to_b, critique_accepted_b_to_a, position_changes_a_after_b, position_changes_b_after_a, a_wins_over_b, b_wins_over_a FROM agent_relationships WHERE agent_a = ? AND agent_b = ?",
            (agent_a, agent_b),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "agent_a": agent_a, "agent_b": agent_b, "debate_count": row[0], "agreement_count": row[1],
            "critique_count_a_to_b": row[2], "critique_count_b_to_a": row[3],
            "critique_accepted_a_to_b": row[4], "critique_accepted_b_to_a": row[5],
            "position_changes_a_after_b": row[6], "position_changes_b_after_a": row[7],
            "a_wins_over_b": row[8], "b_wins_over_a": row[9],
        }

    def get_all_relationships_for_agent(self, agent_name: str) -> list[dict]:
        """Get all relationships involving an agent."""
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:

            cursor = conn.cursor()
        cursor.execute(
            "SELECT agent_a, agent_b, debate_count, agreement_count, critique_count_a_to_b, critique_count_b_to_a, critique_accepted_a_to_b, critique_accepted_b_to_a, position_changes_a_after_b, position_changes_b_after_a, a_wins_over_b, b_wins_over_a FROM agent_relationships WHERE agent_a = ? OR agent_b = ? ORDER BY debate_count DESC",
            (agent_name, agent_name),
        )
        rows = cursor.fetchall()
        return [
            {"agent_a": r[0], "agent_b": r[1], "debate_count": r[2], "agreement_count": r[3],
             "critique_count_a_to_b": r[4], "critique_count_b_to_a": r[5],
             "critique_accepted_a_to_b": r[6], "critique_accepted_b_to_a": r[7],
             "position_changes_a_after_b": r[8], "position_changes_b_after_a": r[9],
             "a_wins_over_b": r[10], "b_wins_over_a": r[11]}
            for r in rows
        ]
