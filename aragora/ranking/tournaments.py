"""
Tournament Management System.

Provides tournament bracket generation, match tracking, and standings
calculation for agent competitions.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TournamentStanding:
    """An agent's standing in a tournament."""
    agent: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    points: float = 0.0
    total_score: float = 0.0

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        total = self.wins + self.losses + self.draws
        if total == 0:
            return 0.0
        return self.wins / total


@dataclass
class TournamentMatch:
    """A match in a tournament."""
    match_id: str
    tournament_id: str
    round_num: int
    agent1: str
    agent2: str
    winner: Optional[str] = None
    score1: float = 0.0
    score2: float = 0.0
    debate_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None


@dataclass
class Tournament:
    """A tournament configuration."""
    tournament_id: str
    name: str
    participants: list[str]
    bracket_type: str = "round_robin"  # round_robin, single_elimination, double_elimination
    rounds_completed: int = 0
    total_rounds: int = 0
    status: str = "pending"  # pending, in_progress, completed
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class TournamentManager:
    """
    Manages tournaments, brackets, and standings.

    Example:
        manager = TournamentManager(db_path="tournaments/contest.db")
        tournament = manager.create_tournament(
            name="Q1 2024 Championship",
            participants=["claude", "gpt4", "gemini"],
            bracket_type="round_robin"
        )
        standings = manager.get_current_standings()
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        nomic_dir: Optional[Path] = None,
    ):
        """
        Initialize TournamentManager.

        Args:
            db_path: Path to tournament SQLite database
            nomic_dir: Base directory for tournament storage
        """
        if db_path:
            self.db_path = Path(db_path)
        elif nomic_dir:
            self.db_path = Path(nomic_dir) / "tournaments" / "default.db"
        else:
            self.db_path = Path("tournaments") / "default.db"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tournaments (
                    tournament_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    bracket_type TEXT DEFAULT 'round_robin',
                    participants TEXT NOT NULL,
                    rounds_completed INTEGER DEFAULT 0,
                    total_rounds INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    match_id TEXT PRIMARY KEY,
                    tournament_id TEXT NOT NULL,
                    round_num INTEGER NOT NULL,
                    agent1 TEXT NOT NULL,
                    agent2 TEXT NOT NULL,
                    winner TEXT,
                    score1 REAL DEFAULT 0.0,
                    score2 REAL DEFAULT 0.0,
                    debate_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT,
                    FOREIGN KEY (tournament_id) REFERENCES tournaments(tournament_id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_matches_tournament
                ON matches(tournament_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_matches_agents
                ON matches(agent1, agent2)
            """)

            conn.commit()
        finally:
            conn.close()

    def create_tournament(
        self,
        name: str,
        participants: list[str],
        bracket_type: str = "round_robin",
    ) -> Tournament:
        """
        Create a new tournament.

        Args:
            name: Tournament name
            participants: List of agent names
            bracket_type: Type of bracket (round_robin, single_elimination, double_elimination)

        Returns:
            Created Tournament object
        """
        tournament_id = f"tournament_{uuid.uuid4().hex[:8]}"

        # Calculate total rounds based on bracket type
        n = len(participants)
        if bracket_type == "round_robin":
            total_rounds = n - 1 if n > 1 else 0
        elif bracket_type == "single_elimination":
            total_rounds = math.ceil(math.log2(n)) if n > 1 else 0
        else:
            total_rounds = n - 1

        tournament = Tournament(
            tournament_id=tournament_id,
            name=name,
            participants=participants,
            bracket_type=bracket_type,
            total_rounds=total_rounds,
            status="pending",
        )

        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                """
                INSERT INTO tournaments
                (tournament_id, name, bracket_type, participants, total_rounds, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tournament.tournament_id,
                    tournament.name,
                    tournament.bracket_type,
                    json.dumps(tournament.participants),
                    tournament.total_rounds,
                    tournament.status,
                    tournament.created_at,
                ),
            )
            conn.commit()
            logger.info(f"Created tournament: {tournament_id}")
        finally:
            conn.close()

        # Generate initial bracket/matches
        self._generate_matches(tournament)

        return tournament

    def _generate_matches(self, tournament: Tournament) -> None:
        """Generate matches for a tournament based on bracket type."""
        if tournament.bracket_type == "round_robin":
            self._generate_round_robin_matches(tournament)
        elif tournament.bracket_type == "single_elimination":
            self._generate_single_elimination_matches(tournament)

    def _generate_round_robin_matches(self, tournament: Tournament) -> None:
        """Generate all matches for round-robin tournament."""
        participants = tournament.participants
        matches = []

        # Round-robin: every participant plays every other participant once
        for i, agent1 in enumerate(participants):
            for agent2 in participants[i + 1:]:
                match = TournamentMatch(
                    match_id=f"match_{uuid.uuid4().hex[:8]}",
                    tournament_id=tournament.tournament_id,
                    round_num=1,  # All in round 1 for round-robin
                    agent1=agent1,
                    agent2=agent2,
                )
                matches.append(match)

        # Insert matches
        conn = sqlite3.connect(str(self.db_path))
        try:
            for match in matches:
                conn.execute(
                    """
                    INSERT INTO matches
                    (match_id, tournament_id, round_num, agent1, agent2, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        match.match_id,
                        match.tournament_id,
                        match.round_num,
                        match.agent1,
                        match.agent2,
                        match.created_at,
                    ),
                )
            conn.commit()
            logger.info(f"Generated {len(matches)} matches for tournament {tournament.tournament_id}")
        finally:
            conn.close()

    def _generate_single_elimination_matches(self, tournament: Tournament) -> None:
        """Generate first round matches for single elimination."""
        participants = tournament.participants
        matches = []

        # First round: pair up participants
        for i in range(0, len(participants) - 1, 2):
            match = TournamentMatch(
                match_id=f"match_{uuid.uuid4().hex[:8]}",
                tournament_id=tournament.tournament_id,
                round_num=1,
                agent1=participants[i],
                agent2=participants[i + 1] if i + 1 < len(participants) else "BYE",
            )
            matches.append(match)

        conn = sqlite3.connect(str(self.db_path))
        try:
            for match in matches:
                conn.execute(
                    """
                    INSERT INTO matches
                    (match_id, tournament_id, round_num, agent1, agent2, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        match.match_id,
                        match.tournament_id,
                        match.round_num,
                        match.agent1,
                        match.agent2,
                        match.created_at,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def record_match_result(
        self,
        match_id: str,
        winner: Optional[str],
        score1: float = 0.0,
        score2: float = 0.0,
        debate_id: Optional[str] = None,
    ) -> None:
        """
        Record the result of a match.

        Args:
            match_id: Match identifier
            winner: Winner agent name (None for draw)
            score1: Score for agent1
            score2: Score for agent2
            debate_id: Optional associated debate ID
        """
        completed_at = datetime.utcnow().isoformat()

        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                """
                UPDATE matches
                SET winner = ?, score1 = ?, score2 = ?, debate_id = ?, completed_at = ?
                WHERE match_id = ?
                """,
                (winner, score1, score2, debate_id, completed_at, match_id),
            )
            conn.commit()
            logger.info(f"Recorded result for match {match_id}: winner={winner}")
        finally:
            conn.close()

    def get_current_standings(self) -> list[TournamentStanding]:
        """
        Get current tournament standings.

        Returns:
            List of TournamentStanding sorted by points (descending)
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            # Get all participants from tournaments
            cursor = conn.execute("SELECT participants FROM tournaments LIMIT 1")
            row = cursor.fetchone()
            if not row:
                return []

            participants = json.loads(row[0])
            standings_dict = {p: TournamentStanding(agent=p) for p in participants}

            # Calculate standings from completed matches
            cursor = conn.execute(
                """
                SELECT agent1, agent2, winner, score1, score2
                FROM matches
                WHERE completed_at IS NOT NULL
                """
            )

            for row in cursor:
                agent1, agent2, winner, score1, score2 = row

                if agent1 in standings_dict:
                    standings_dict[agent1].total_score += score1

                if agent2 in standings_dict:
                    standings_dict[agent2].total_score += score2

                if winner is None:
                    # Draw
                    if agent1 in standings_dict:
                        standings_dict[agent1].draws += 1
                        standings_dict[agent1].points += 0.5
                    if agent2 in standings_dict:
                        standings_dict[agent2].draws += 1
                        standings_dict[agent2].points += 0.5
                elif winner == agent1:
                    if agent1 in standings_dict:
                        standings_dict[agent1].wins += 1
                        standings_dict[agent1].points += 1.0
                    if agent2 in standings_dict:
                        standings_dict[agent2].losses += 1
                elif winner == agent2:
                    if agent2 in standings_dict:
                        standings_dict[agent2].wins += 1
                        standings_dict[agent2].points += 1.0
                    if agent1 in standings_dict:
                        standings_dict[agent1].losses += 1

            # Sort by points (descending), then win rate
            standings = list(standings_dict.values())
            standings.sort(key=lambda s: (s.points, s.win_rate), reverse=True)

            return standings

        finally:
            conn.close()

    def get_matches(
        self,
        tournament_id: Optional[str] = None,
        round_num: Optional[int] = None,
        completed_only: bool = False,
    ) -> list[TournamentMatch]:
        """
        Get matches with optional filtering.

        Args:
            tournament_id: Filter by tournament
            round_num: Filter by round number
            completed_only: Only return completed matches

        Returns:
            List of TournamentMatch objects
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            query = "SELECT * FROM matches WHERE 1=1"
            params = []

            if tournament_id:
                query += " AND tournament_id = ?"
                params.append(tournament_id)

            if round_num is not None:
                query += " AND round_num = ?"
                params.append(round_num)

            if completed_only:
                query += " AND completed_at IS NOT NULL"

            query += " ORDER BY created_at DESC"

            cursor = conn.execute(query, params)
            matches = []

            for row in cursor:
                matches.append(TournamentMatch(
                    match_id=row[0],
                    tournament_id=row[1],
                    round_num=row[2],
                    agent1=row[3],
                    agent2=row[4],
                    winner=row[5],
                    score1=row[6] or 0.0,
                    score2=row[7] or 0.0,
                    debate_id=row[8],
                    created_at=row[9],
                    completed_at=row[10],
                ))

            return matches

        finally:
            conn.close()

    def get_tournament(self, tournament_id: str) -> Optional[Tournament]:
        """Get tournament by ID."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute(
                "SELECT * FROM tournaments WHERE tournament_id = ?",
                (tournament_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            return Tournament(
                tournament_id=row[0],
                name=row[1],
                bracket_type=row[2],
                participants=json.loads(row[3]),
                rounds_completed=row[4],
                total_rounds=row[5],
                status=row[6],
                created_at=row[7],
            )
        finally:
            conn.close()

    def list_tournaments(self, limit: int = 50) -> list[Tournament]:
        """List all tournaments."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute(
                "SELECT * FROM tournaments ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            tournaments = []

            for row in cursor:
                tournaments.append(Tournament(
                    tournament_id=row[0],
                    name=row[1],
                    bracket_type=row[2],
                    participants=json.loads(row[3]),
                    rounds_completed=row[4],
                    total_rounds=row[5],
                    status=row[6],
                    created_at=row[7],
                ))

            return tournaments

        finally:
            conn.close()

    def advance_round(self, tournament_id: str) -> bool:
        """
        Advance tournament to next round (for elimination brackets).

        Returns:
            True if advanced, False if tournament is complete
        """
        tournament = self.get_tournament(tournament_id)
        if not tournament:
            return False

        if tournament.bracket_type != "single_elimination":
            # Round-robin doesn't need advancement
            return False

        # Check if all matches in current round are complete
        matches = self.get_matches(tournament_id=tournament_id)
        current_round = tournament.rounds_completed + 1
        current_round_matches = [m for m in matches if m.round_num == current_round]

        if not all(m.completed_at for m in current_round_matches):
            logger.warning(f"Cannot advance: round {current_round} not complete")
            return False

        # Get winners for next round
        winners = [m.winner for m in current_round_matches if m.winner]

        if len(winners) <= 1:
            # Tournament complete
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.execute(
                    "UPDATE tournaments SET status = 'completed', rounds_completed = ? WHERE tournament_id = ?",
                    (current_round, tournament_id),
                )
                conn.commit()
            finally:
                conn.close()
            return False

        # Generate next round matches
        next_round = current_round + 1
        new_matches = []

        for i in range(0, len(winners) - 1, 2):
            match = TournamentMatch(
                match_id=f"match_{uuid.uuid4().hex[:8]}",
                tournament_id=tournament_id,
                round_num=next_round,
                agent1=winners[i],
                agent2=winners[i + 1] if i + 1 < len(winners) else "BYE",
            )
            new_matches.append(match)

        conn = sqlite3.connect(str(self.db_path))
        try:
            for match in new_matches:
                conn.execute(
                    """
                    INSERT INTO matches
                    (match_id, tournament_id, round_num, agent1, agent2, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        match.match_id,
                        match.tournament_id,
                        match.round_num,
                        match.agent1,
                        match.agent2,
                        match.created_at,
                    ),
                )

            conn.execute(
                "UPDATE tournaments SET rounds_completed = ?, status = 'in_progress' WHERE tournament_id = ?",
                (current_round, tournament_id),
            )
            conn.commit()
            logger.info(f"Advanced tournament {tournament_id} to round {next_round}")
        finally:
            conn.close()

        return True
