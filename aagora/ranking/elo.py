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

    def __init__(self, db_path: str = "aagora_elo.db"):
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

        conn.commit()
        conn.close()

    def get_rating(self, agent_name: str) -> AgentRating:
        """Get or create rating for an agent."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT agent_name, elo, domain_elos, wins, losses, draws,
                   debates_count, critiques_accepted, critiques_total, updated_at
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
            updated_at=row[9],
        )

    def _save_rating(self, rating: AgentRating):
        """Save rating to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO ratings (agent_name, elo, domain_elos, wins, losses, draws,
                                debates_count, critiques_accepted, critiques_total, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_name) DO UPDATE SET
                elo = excluded.elo,
                domain_elos = excluded.domain_elos,
                wins = excluded.wins,
                losses = excluded.losses,
                draws = excluded.draws,
                debates_count = excluded.debates_count,
                critiques_accepted = excluded.critiques_accepted,
                critiques_total = excluded.critiques_total,
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
