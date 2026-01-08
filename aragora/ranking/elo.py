"""
ELO/Reputation System for agent skill tracking.

Inspired by ChatArena's competitive environments, this module provides:
- ELO ratings for agents
- Domain-specific skill ratings
- Match history and statistics
- Leaderboards

Performance: Uses LRU caching for frequently accessed data like leaderboards.
Cache is automatically invalidated when ratings are updated.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from aragora.storage.schema import SchemaManager, safe_add_column
from aragora.config import (
    DB_ELO_PATH,
    DB_TIMEOUT_SECONDS,
    ELO_INITIAL_RATING,
    ELO_K_FACTOR,
    ELO_CALIBRATION_MIN_COUNT,
    CACHE_TTL_LEADERBOARD,
    CACHE_TTL_RECENT_MATCHES,
    CACHE_TTL_LB_STATS,
    CACHE_TTL_CALIBRATION_LB,
)
from aragora.ranking.database import EloDatabase
from aragora.ranking.relationships import RelationshipTracker, RelationshipStats, RelationshipMetrics
from aragora.ranking.redteam import RedTeamIntegrator, RedTeamResult, VulnerabilitySummary
from aragora.ranking.elo_core import (
    expected_score,
    calculate_new_elo,
    calculate_pairwise_elo_changes,
    apply_elo_changes,
)
from aragora.utils.json_helpers import safe_json_loads
from aragora.utils.cache import TTLCache, lru_cache_with_ttl

logger = logging.getLogger(__name__)

# Re-export for backwards compatibility
__all__ = [
    "EloSystem",
    "AgentRating",
    "MatchResult",
    "RelationshipTracker",
    "RelationshipStats",
    "RelationshipMetrics",
    "RedTeamIntegrator",
    "RedTeamResult",
    "VulnerabilitySummary",
]

# Schema version - increment when making schema changes
ELO_SCHEMA_VERSION = 2

# Use centralized config values (can be overridden via environment variables)
DEFAULT_ELO = ELO_INITIAL_RATING
K_FACTOR = ELO_K_FACTOR
CALIBRATION_MIN_COUNT = ELO_CALIBRATION_MIN_COUNT


def _escape_like_pattern(value: str) -> str:
    """Escape special characters in SQL LIKE patterns to prevent injection."""
    return value.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')


# Maximum agent name length (matches SAFE_AGENT_PATTERN in validation/entities.py)
MAX_AGENT_NAME_LENGTH = 32


def _validate_agent_name(agent_name: str) -> None:
    """Validate agent name length to prevent performance issues.

    Args:
        agent_name: Agent name to validate

    Raises:
        ValueError: If agent name exceeds MAX_AGENT_NAME_LENGTH
    """
    if len(agent_name) > MAX_AGENT_NAME_LENGTH:
        raise ValueError(
            f"Agent name exceeds {MAX_AGENT_NAME_LENGTH} characters: {len(agent_name)}"
        )


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
    Uses LRU caching for frequently accessed data.
    """

    # Class-level cache for leaderboard data (shared across instances)
    _leaderboard_cache: TTLCache[list] = TTLCache(maxsize=50, ttl_seconds=CACHE_TTL_LEADERBOARD)
    _rating_cache: TTLCache[AgentRating] = TTLCache(maxsize=200, ttl_seconds=CACHE_TTL_RECENT_MATCHES)
    _stats_cache: TTLCache[dict] = TTLCache(maxsize=10, ttl_seconds=CACHE_TTL_LB_STATS)
    _calibration_cache: TTLCache[list] = TTLCache(maxsize=20, ttl_seconds=CACHE_TTL_CALIBRATION_LB)

    def __init__(self, db_path: str = DB_ELO_PATH):
        self.db_path = Path(db_path)
        self._db = EloDatabase(db_path)
        self._init_db()

        # Delegate to extracted modules (lazy initialization)
        self._relationship_tracker: RelationshipTracker | None = None
        self._redteam_integrator: RedTeamIntegrator | None = None

    @property
    def relationship_tracker(self) -> RelationshipTracker:
        """Get the relationship tracker (lazy initialized)."""
        if self._relationship_tracker is None:
            self._relationship_tracker = RelationshipTracker(self.db_path)
        return self._relationship_tracker

    @property
    def redteam_integrator(self) -> RedTeamIntegrator:
        """Get the red team integrator (lazy initialized)."""
        if self._redteam_integrator is None:
            self._redteam_integrator = RedTeamIntegrator(self)
        return self._redteam_integrator

    def _init_db(self) -> None:
        """Initialize database schema using SchemaManager."""
        with self._db.connection() as conn:
            manager = SchemaManager(conn, "elo", current_version=ELO_SCHEMA_VERSION)

            # Register migration from v1 to v2: add calibration columns
            manager.register_migration(
                from_version=1,
                to_version=2,
                function=self._migrate_v1_to_v2,
                description="Add calibration columns to ratings table",
            )

            # Initial schema (v1)
            initial_schema = """
                -- Agent ratings
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
                );

                -- Match history
                CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    debate_id TEXT UNIQUE,
                    winner TEXT,
                    participants TEXT,
                    domain TEXT,
                    scores TEXT,
                    elo_changes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- ELO history for tracking progression
                CREATE TABLE IF NOT EXISTS elo_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    elo REAL NOT NULL,
                    debate_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Calibration predictions table
                CREATE TABLE IF NOT EXISTS calibration_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tournament_id TEXT NOT NULL,
                    predictor_agent TEXT NOT NULL,
                    predicted_winner TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(tournament_id, predictor_agent)
                );

                -- Domain-specific calibration tracking
                CREATE TABLE IF NOT EXISTS domain_calibration (
                    agent_name TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    total_predictions INTEGER DEFAULT 0,
                    total_correct INTEGER DEFAULT 0,
                    brier_sum REAL DEFAULT 0.0,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (agent_name, domain)
                );

                -- Calibration by confidence bucket
                CREATE TABLE IF NOT EXISTS calibration_buckets (
                    agent_name TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    bucket_key TEXT NOT NULL,
                    predictions INTEGER DEFAULT 0,
                    correct INTEGER DEFAULT 0,
                    brier_sum REAL DEFAULT 0.0,
                    PRIMARY KEY (agent_name, domain, bucket_key)
                );

                -- Agent relationships tracking
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
                );

                -- Performance indexes
                CREATE INDEX IF NOT EXISTS idx_elo_history_agent ON elo_history(agent_name);
                CREATE INDEX IF NOT EXISTS idx_elo_history_created ON elo_history(created_at);
                CREATE INDEX IF NOT EXISTS idx_elo_history_debate ON elo_history(debate_id);
                CREATE INDEX IF NOT EXISTS idx_matches_winner ON matches(winner);
                CREATE INDEX IF NOT EXISTS idx_matches_created ON matches(created_at);
                CREATE INDEX IF NOT EXISTS idx_matches_domain ON matches(domain);
                CREATE INDEX IF NOT EXISTS idx_domain_cal_agent ON domain_calibration(agent_name);
                CREATE INDEX IF NOT EXISTS idx_domain_cal_agent_domain ON domain_calibration(agent_name, domain);
                CREATE INDEX IF NOT EXISTS idx_relationships_a ON agent_relationships(agent_a);
                CREATE INDEX IF NOT EXISTS idx_relationships_b ON agent_relationships(agent_b);
                CREATE INDEX IF NOT EXISTS idx_calibration_pred_tournament ON calibration_predictions(tournament_id);
                CREATE INDEX IF NOT EXISTS idx_ratings_agent ON ratings(agent_name);
            """

            manager.ensure_schema(initial_schema=initial_schema)

    def _migrate_v1_to_v2(self, conn: sqlite3.Connection) -> None:
        """Migration: Add calibration columns to ratings table."""
        safe_add_column(conn, "ratings", "calibration_correct", "INTEGER", default="0")
        safe_add_column(conn, "ratings", "calibration_total", "INTEGER", default="0")
        safe_add_column(conn, "ratings", "calibration_brier_sum", "REAL", default="0.0")


    def get_rating(self, agent_name: str, use_cache: bool = True) -> AgentRating:
        """Get or create rating for an agent.

        Args:
            agent_name: Name of the agent
            use_cache: Whether to use cached value (default True). Set False
                      for operations that need the latest data.
        """
        _validate_agent_name(agent_name)
        cache_key = f"rating:{agent_name}"

        # Check cache first
        if use_cache:
            cached = self._rating_cache.get(cache_key)
            if cached is not None:
                return cached

        with self._db.connection() as conn:
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
            rating = AgentRating(agent_name=agent_name)
        else:
            rating = AgentRating(
                agent_name=row[0],
                elo=row[1],
                domain_elos=safe_json_loads(row[2], {}),
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

        # Cache the result
        self._rating_cache.set(cache_key, rating)
        return rating

    def get_ratings_batch(self, agent_names: list[str]) -> dict[str, AgentRating]:
        """Get ratings for multiple agents in a single query (batch optimization).

        Args:
            agent_names: List of agent names to fetch ratings for

        Returns:
            Dict mapping agent_name -> AgentRating. Missing agents get default ratings.
        """
        if not agent_names:
            return {}

        result = {}
        with self._db.connection() as conn:
            cursor = conn.cursor()

            # Use parameterized IN clause
            placeholders = ','.join('?' * len(agent_names))
            cursor.execute(
                f"""
                SELECT agent_name, elo, domain_elos, wins, losses, draws,
                       debates_count, critiques_accepted, critiques_total,
                       calibration_correct, calibration_total, calibration_brier_sum,
                       updated_at
                FROM ratings WHERE agent_name IN ({placeholders})
                """,
                tuple(agent_names),
            )
            rows = cursor.fetchall()

        # Build result dict from fetched rows
        for row in rows:
            rating = AgentRating(
                agent_name=row[0],
                elo=row[1],
                domain_elos=safe_json_loads(row[2], {}),
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
            result[rating.agent_name] = rating

        # Add default ratings for agents not found in DB
        for name in agent_names:
            if name not in result:
                result[name] = AgentRating(agent_name=name)

        return result

    def list_agents(self) -> list[str]:
        """Get list of all known agent names.

        Returns:
            List of agent names that have ratings recorded.
        """
        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT agent_name FROM ratings ORDER BY elo DESC")
            return [row[0] for row in cursor.fetchall()]

    def get_all_ratings(self) -> list[AgentRating]:
        """Get all agent ratings in a single query (batch optimization).

        More efficient than calling get_rating() for each agent.

        Returns:
            List of all AgentRating objects, sorted by ELO descending.
        """
        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT agent_name, elo, domain_elos, wins, losses, draws,
                       debates_count, critiques_accepted, critiques_total,
                       calibration_correct, calibration_total, calibration_brier_sum,
                       updated_at
                FROM ratings
                ORDER BY elo DESC
                """
            )
            rows = cursor.fetchall()

        return [
            AgentRating(
                agent_name=row[0],
                elo=row[1],
                domain_elos=safe_json_loads(row[2], {}),
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

    def _save_rating(self, rating: AgentRating) -> None:
        """Save rating to database."""
        with self._db.connection() as conn:
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

        # Invalidate caches after write
        self._rating_cache.invalidate(f"rating:{rating.agent_name}")
        self._leaderboard_cache.clear()
        self._stats_cache.clear()
        self._calibration_cache.clear()

    def _save_ratings_batch(self, ratings: list[AgentRating]) -> None:
        """Save multiple ratings in a single transaction.

        More efficient than calling _save_rating() in a loop.
        """
        if not ratings:
            return

        with self._db.connection() as conn:
            cursor = conn.cursor()
            for rating in ratings:
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

        # Invalidate caches after batch write
        for rating in ratings:
            self._rating_cache.invalidate(f"rating:{rating.agent_name}")
        self._leaderboard_cache.clear()
        self._stats_cache.clear()
        self._calibration_cache.clear()

    def _record_elo_history_batch(
        self, entries: list[tuple[str, float, str | None]]
    ) -> None:
        """Record multiple ELO history entries in a single transaction.

        Args:
            entries: List of (agent_name, elo, debate_id) tuples
        """
        if not entries:
            return

        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                "INSERT INTO elo_history (agent_name, elo, debate_id) VALUES (?, ?, ?)",
                entries,
            )
            conn.commit()

    # Core ELO calculations delegated to elo_core module
    # These methods are kept for backward compatibility but delegate to pure functions

    def _expected_score(self, elo_a: float, elo_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return expected_score(elo_a, elo_b)

    def _calculate_new_elo(
        self,
        current_elo: float,
        expected: float,
        actual: float,
        k: float = K_FACTOR,
    ) -> float:
        """Calculate new ELO rating."""
        return calculate_new_elo(current_elo, expected, actual, k)

    def _calculate_pairwise_elo_changes(
        self,
        participants: list[str],
        scores: dict[str, float],
        ratings: dict[str, "AgentRating"],
        confidence_weight: float,
    ) -> dict[str, float]:
        """Calculate pairwise ELO changes for all participant combinations."""
        return calculate_pairwise_elo_changes(
            participants, scores, ratings, confidence_weight, K_FACTOR
        )

    def _apply_elo_changes(
        self,
        elo_changes: dict[str, float],
        ratings: dict[str, "AgentRating"],
        winner: Optional[str],
        domain: Optional[str],
        debate_id: str,
    ) -> tuple[list["AgentRating"], list[tuple[str, float, str]]]:
        """Apply ELO changes to ratings and prepare for batch save."""
        return apply_elo_changes(
            elo_changes, ratings, winner, domain, debate_id, DEFAULT_ELO
        )

    def record_match(
        self,
        debate_id: str,
        participants: list[str],
        scores: dict[str, float],
        domain: str | None = None,
        confidence_weight: float = 1.0,
    ) -> dict[str, float]:
        """
        Record a match result and update ELO ratings.

        Args:
            debate_id: Unique debate identifier
            participants: List of agent names
            scores: Dict of agent -> score (higher is better)
            domain: Optional domain for domain-specific ELO
            confidence_weight: Weight for ELO change (0-1). Lower values reduce
                               ELO impact for low-confidence debates. Default 1.0.

        Returns:
            Dict of agent -> ELO change
        """
        # Clamp confidence_weight to valid range
        confidence_weight = max(0.1, min(1.0, confidence_weight))
        if len(participants) < 2:
            return {}

        # Determine winner (highest score)
        sorted_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_agents) < 2:
            winner = sorted_agents[0][0] if sorted_agents else None
        else:
            winner = sorted_agents[0][0] if sorted_agents[0][1] > sorted_agents[1][1] else None

        # Get current ratings (batch query to avoid N+1)
        ratings = self.get_ratings_batch(participants)

        # Calculate pairwise ELO changes
        elo_changes = self._calculate_pairwise_elo_changes(
            participants, scores, ratings, confidence_weight
        )

        # Apply changes and collect for batch save
        ratings_to_save, history_entries = self._apply_elo_changes(
            elo_changes, ratings, winner, domain, debate_id
        )

        # Batch save all ratings and history (single transaction each)
        self._save_ratings_batch(ratings_to_save)
        self._record_elo_history_batch(history_entries)

        # Save match
        self._save_match(debate_id, winner, participants, domain, scores, elo_changes)

        # Write JSON snapshot for fast reads (avoids SQLite locking)
        self._write_snapshot()

        # Invalidate related caches so API returns fresh data
        try:
            from aragora.server.handlers.base import invalidate_on_event
            invalidate_on_event("match_recorded")
        except ImportError:
            pass  # Handlers may not be available in all contexts

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
        with self._db.connection() as conn:
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

    def _record_elo_history(self, agent_name: str, elo: float, debate_id: str | None = None) -> None:
        """Record ELO at a point in time."""
        with self._db.connection() as conn:
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
        except Exception as e:
            # Snapshot is optional, don't fail on write errors
            logger.debug(f"Failed to write ELO snapshot: {e}")
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
            except json.JSONDecodeError as e:
                logger.debug(f"ELO snapshot corrupted, falling back to database: {e}")
            except (PermissionError, OSError) as e:
                logger.warning(f"Cannot read ELO snapshot (I/O error), falling back to database: {e}")

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
            except json.JSONDecodeError as e:
                logger.debug(f"Recent matches snapshot corrupted, falling back to database: {e}")
            except (PermissionError, OSError) as e:
                logger.warning(f"Cannot read recent matches snapshot (I/O error), falling back: {e}")

        # Fall back to database
        return self.get_recent_matches(limit)

    def record_critique(self, agent_name: str, accepted: bool) -> None:
        """Record a critique and whether it was accepted."""
        rating = self.get_rating(agent_name)
        rating.critiques_total += 1
        if accepted:
            rating.critiques_accepted += 1
        rating.updated_at = datetime.now().isoformat()
        self._save_rating(rating)

    def get_leaderboard(self, limit: int = 20, domain: str | None = None) -> list[AgentRating]:
        """Get top agents by ELO.

        Args:
            limit: Maximum number of agents to return
            domain: Optional domain to sort by. If provided, sorts by domain-specific ELO
                   using SQLite JSON extraction for efficiency.
        """
        with self._db.connection() as conn:
            cursor = conn.cursor()

            if domain:
                # Use SQLite JSON extraction for domain-specific leaderboard
                # COALESCE handles missing domain entries by falling back to global ELO
                cursor.execute(
                    """
                    SELECT agent_name, elo, domain_elos, wins, losses, draws,
                           debates_count, critiques_accepted, critiques_total, updated_at
                    FROM ratings
                    ORDER BY COALESCE(json_extract(domain_elos, ?), elo) DESC
                    LIMIT ?
                    """,
                    (f'$."{domain}"', limit),
                )
            else:
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

        return [
            AgentRating(
                agent_name=row[0],
                elo=row[1],
                domain_elos=safe_json_loads(row[2], {}),
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

    def get_cached_leaderboard(
        self, limit: int = 20, domain: str | None = None
    ) -> list[AgentRating]:
        """
        Get leaderboard with caching for better performance.

        Uses a 5-minute TTL cache. For real-time data, use get_leaderboard() directly.

        Args:
            limit: Maximum number of agents to return
            domain: Optional domain to sort by

        Returns:
            Cached list of AgentRating sorted by ELO
        """
        cache_key = f"leaderboard:{limit}:{domain or 'global'}"
        cached = self._leaderboard_cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Leaderboard cache hit for {cache_key}")
            return cached

        # Cache miss - fetch from database
        result = self.get_leaderboard(limit=limit, domain=domain)
        self._leaderboard_cache.set(cache_key, result)
        logger.debug(f"Leaderboard cache miss, stored {cache_key}")
        return result

    def invalidate_leaderboard_cache(self) -> int:
        """Invalidate all cached leaderboard data. Call after rating changes."""
        self._stats_cache.clear()
        self._calibration_cache.clear()
        return self._leaderboard_cache.clear()

    def invalidate_rating_cache(self, agent_name: str | None = None) -> int:
        """Invalidate cached ratings. Pass agent_name for specific agent, None for all."""
        if agent_name:
            return 1 if self._rating_cache.invalidate(f"rating:{agent_name}") else 0
        return self._rating_cache.clear()

    def get_top_agents_for_domain(self, domain: str, limit: int = 5) -> list[AgentRating]:
        """Get agents ranked by domain-specific performance.

        Args:
            domain: Domain to rank by (e.g., 'security', 'performance', 'architecture')
            limit: Maximum number of agents to return

        Returns:
            List of AgentRating sorted by domain-specific ELO (highest first)
        """
        return self.get_cached_leaderboard(limit=limit, domain=domain)

    def get_elo_history(self, agent_name: str, limit: int = 50) -> list[tuple[str, float]]:
        """Get ELO history for an agent."""
        with self._db.connection() as conn:
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
        with self._db.connection() as conn:
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
            elo_changes = safe_json_loads(row[4], {})
            participants = safe_json_loads(row[2], [])
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
        _validate_agent_name(agent_a)
        _validate_agent_name(agent_b)

        with self._db.connection() as conn:
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

    def get_stats(self, use_cache: bool = True) -> dict:
        """Get overall system statistics.

        Args:
            use_cache: Whether to use cached value (default True).

        Returns:
            Dict with total_agents, avg_elo, max_elo, min_elo, total_matches.
        """
        cache_key = "elo_stats"

        if use_cache:
            cached = self._stats_cache.get(cache_key)
            if cached is not None:
                return cached

        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), AVG(elo), MAX(elo), MIN(elo) FROM ratings")
            ratings_row = cursor.fetchone()
            cursor.execute("SELECT COUNT(*) FROM matches")
            matches_row = cursor.fetchone()

        # Handle case where fetchone returns None
        if ratings_row is None:
            ratings_row = (0, None, None, None)
        if matches_row is None:
            matches_row = (0,)

        result = {
            "total_agents": ratings_row[0] or 0,
            "avg_elo": ratings_row[1] or DEFAULT_ELO,
            "max_elo": ratings_row[2] or DEFAULT_ELO,
            "min_elo": ratings_row[3] or DEFAULT_ELO,
            "total_matches": matches_row[0] or 0,
        }

        self._stats_cache.set(cache_key, result)
        return result

    # =========================================================================
    # Tournament Winner Calibration Scoring
    # =========================================================================

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
        with self._db.connection() as conn:
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
        with self._db.connection() as conn:
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

        # Batch load all predictor ratings upfront (avoids N+1 query)
        predictor_names = [p[0] for p in predictions]
        ratings = self.get_ratings_batch(predictor_names)

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
        self._save_ratings_batch(list(ratings.values()))

        return brier_scores

    def get_calibration_leaderboard(self, limit: int = 20, use_cache: bool = True) -> list[AgentRating]:
        """
        Get agents ranked by calibration score.

        Only includes agents with minimum predictions.

        Args:
            limit: Maximum number of agents to return
            use_cache: Whether to use cached value (default True)
        """
        cache_key = f"calibration_lb:{limit}"

        if use_cache:
            cached = self._calibration_cache.get(cache_key)
            if cached is not None:
                return cached

        with self._db.connection() as conn:
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

        result = [
            AgentRating(
                agent_name=row[0],
                elo=row[1],
                domain_elos=safe_json_loads(row[2], {}),
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

        self._calibration_cache.set(cache_key, result)
        return result

    def get_agent_calibration_history(
        self, agent_name: str, limit: int = 50
    ) -> list[dict]:
        """Get recent predictions made by an agent."""
        with self._db.connection() as conn:
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
    ) -> None:
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

        with self._db.connection() as conn:
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
        with self._db.connection() as conn:
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
        with self._db.connection() as conn:
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
    # Delegated to RelationshipTracker for cleaner separation of concerns.
    # These methods are kept for backwards compatibility.
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
    ) -> None:
        """Update relationship stats between two agents.

        Delegates to RelationshipTracker. For new code, use:
            elo_system.relationship_tracker.update_relationship(...)
        """
        self.relationship_tracker.update_relationship(
            agent_a=agent_a,
            agent_b=agent_b,
            debate_increment=debate_increment,
            agreement_increment=agreement_increment,
            critique_a_to_b=critique_a_to_b,
            critique_b_to_a=critique_b_to_a,
            critique_accepted_a_to_b=critique_accepted_a_to_b,
            critique_accepted_b_to_a=critique_accepted_b_to_a,
            position_change_a_after_b=position_change_a_after_b,
            position_change_b_after_a=position_change_b_after_a,
            a_win=a_win,
            b_win=b_win,
        )

    def update_relationships_batch(self, updates: list[dict]) -> None:
        """Batch update multiple agent relationships.

        Delegates to RelationshipTracker. For new code, use:
            elo_system.relationship_tracker.update_batch(...)
        """
        self.relationship_tracker.update_batch(updates)

    def get_relationship_raw(self, agent_a: str, agent_b: str) -> Optional[dict]:
        """Get raw relationship data between two agents.

        Delegates to RelationshipTracker. For new code, use:
            elo_system.relationship_tracker.get_raw(...)
        """
        stats = self.relationship_tracker.get_raw(agent_a, agent_b)
        if stats is None:
            return None
        # Convert dataclass to dict for backwards compatibility
        return {
            "agent_a": stats.agent_a,
            "agent_b": stats.agent_b,
            "debate_count": stats.debate_count,
            "agreement_count": stats.agreement_count,
            "critique_count_a_to_b": stats.critique_count_a_to_b,
            "critique_count_b_to_a": stats.critique_count_b_to_a,
            "critique_accepted_a_to_b": stats.critique_accepted_a_to_b,
            "critique_accepted_b_to_a": stats.critique_accepted_b_to_a,
            "position_changes_a_after_b": stats.position_changes_a_after_b,
            "position_changes_b_after_a": stats.position_changes_b_after_a,
            "a_wins_over_b": stats.a_wins_over_b,
            "b_wins_over_a": stats.b_wins_over_a,
        }

    def get_all_relationships_for_agent(self, agent_name: str, limit: int = 100) -> list[dict]:
        """Get all relationships involving an agent.

        Delegates to RelationshipTracker. For new code, use:
            elo_system.relationship_tracker.get_all_for_agent(...)
        """
        _validate_agent_name(agent_name)
        stats_list = self.relationship_tracker.get_all_for_agent(agent_name, limit)
        # Convert dataclasses to dicts for backwards compatibility
        return [
            {
                "agent_a": s.agent_a,
                "agent_b": s.agent_b,
                "debate_count": s.debate_count,
                "agreement_count": s.agreement_count,
                "critique_count_a_to_b": s.critique_count_a_to_b,
                "critique_count_b_to_a": s.critique_count_b_to_a,
                "critique_accepted_a_to_b": s.critique_accepted_a_to_b,
                "critique_accepted_b_to_a": s.critique_accepted_b_to_a,
                "position_changes_a_after_b": s.position_changes_a_after_b,
                "position_changes_b_after_a": s.position_changes_b_after_a,
                "a_wins_over_b": s.a_wins_over_b,
                "b_wins_over_a": s.b_wins_over_a,
            }
            for s in stats_list
        ]

    def compute_relationship_metrics(self, agent_a: str, agent_b: str) -> dict:
        """Compute rivalry and alliance scores between two agents.

        Delegates to RelationshipTracker. For new code, use:
            elo_system.relationship_tracker.compute_metrics(...)
        """
        metrics = self.relationship_tracker.compute_metrics(agent_a, agent_b)
        # Convert dataclass to dict for backwards compatibility
        return {
            "agent_a": metrics.agent_a,
            "agent_b": metrics.agent_b,
            "rivalry_score": metrics.rivalry_score,
            "alliance_score": metrics.alliance_score,
            "relationship": metrics.relationship,
            "debate_count": metrics.debate_count,
            "agreement_rate": metrics.agreement_rate,
            "head_to_head": metrics.head_to_head,
        }

    def _compute_metrics_from_raw(self, agent_a: str, agent_b: str, raw: dict) -> dict:
        """Compute relationship metrics from raw data (no database call).

        For backwards compatibility. New code should use RelationshipTracker.
        """
        # Create a RelationshipStats from the raw dict
        stats = RelationshipStats(
            agent_a=raw.get("agent_a", agent_a),
            agent_b=raw.get("agent_b", agent_b),
            debate_count=raw.get("debate_count", 0),
            agreement_count=raw.get("agreement_count", 0),
            critique_count_a_to_b=raw.get("critique_count_a_to_b", 0),
            critique_count_b_to_a=raw.get("critique_count_b_to_a", 0),
            critique_accepted_a_to_b=raw.get("critique_accepted_a_to_b", 0),
            critique_accepted_b_to_a=raw.get("critique_accepted_b_to_a", 0),
            position_changes_a_after_b=raw.get("position_changes_a_after_b", 0),
            position_changes_b_after_a=raw.get("position_changes_b_after_a", 0),
            a_wins_over_b=raw.get("a_wins_over_b", 0),
            b_wins_over_a=raw.get("b_wins_over_a", 0),
        )
        metrics = self.relationship_tracker._compute_metrics_from_stats(agent_a, agent_b, stats)
        return {
            "agent_a": metrics.agent_a,
            "agent_b": metrics.agent_b,
            "rivalry_score": metrics.rivalry_score,
            "alliance_score": metrics.alliance_score,
            "relationship": metrics.relationship,
            "debate_count": metrics.debate_count,
        }

    def get_rivals(self, agent_name: str, limit: int = 5) -> list[dict]:
        """Get agent's top rivals by rivalry score.

        Delegates to RelationshipTracker. For new code, use:
            elo_system.relationship_tracker.get_rivals(...)
        """
        _validate_agent_name(agent_name)
        metrics_list = self.relationship_tracker.get_rivals(agent_name, limit)
        return [
            {
                "agent_a": m.agent_a,
                "agent_b": m.agent_b,
                "rivalry_score": m.rivalry_score,
                "alliance_score": m.alliance_score,
                "relationship": m.relationship,
                "debate_count": m.debate_count,
            }
            for m in metrics_list
        ]

    def get_allies(self, agent_name: str, limit: int = 5) -> list[dict]:
        """Get agent's top allies by alliance score.

        Delegates to RelationshipTracker. For new code, use:
            elo_system.relationship_tracker.get_allies(...)
        """
        _validate_agent_name(agent_name)
        metrics_list = self.relationship_tracker.get_allies(agent_name, limit)
        return [
            {
                "agent_a": m.agent_a,
                "agent_b": m.agent_b,
                "rivalry_score": m.rivalry_score,
                "alliance_score": m.alliance_score,
                "relationship": m.relationship,
                "debate_count": m.debate_count,
            }
            for m in metrics_list
        ]

    # =========================================================================
    # Red Team Integration (Vulnerability-based ELO adjustment)
    # Delegated to RedTeamIntegrator for cleaner separation of concerns.
    # These methods are kept for backwards compatibility.
    # =========================================================================

    def record_redteam_result(
        self,
        agent_name: str,
        robustness_score: float,
        successful_attacks: int,
        total_attacks: int,
        critical_vulnerabilities: int = 0,
        session_id: str | None = None,
    ) -> float:
        """Record red team results and adjust ELO based on vulnerability.

        Delegates to RedTeamIntegrator. For new code, use:
            elo_system.redteam_integrator.record_result(...)
        """
        return self.redteam_integrator.record_result(
            agent_name=agent_name,
            robustness_score=robustness_score,
            successful_attacks=successful_attacks,
            total_attacks=total_attacks,
            critical_vulnerabilities=critical_vulnerabilities,
            session_id=session_id,
        )

    def get_vulnerability_summary(self, agent_name: str) -> dict:
        """Get summary of agent's red team history.

        Delegates to RedTeamIntegrator. For new code, use:
            elo_system.redteam_integrator.get_vulnerability_summary(...)
        """
        summary = self.redteam_integrator.get_vulnerability_summary(agent_name)
        # Convert dataclass to dict for backwards compatibility
        return {
            "redteam_sessions": summary.redteam_sessions,
            "total_elo_impact": summary.total_elo_impact,
            "last_session": summary.last_session,
        }
