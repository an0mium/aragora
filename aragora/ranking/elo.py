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
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.typing import EventEmitterProtocol

from aragora.config import (
    CACHE_TTL_CALIBRATION_LB,
    CACHE_TTL_LB_STATS,
    CACHE_TTL_LEADERBOARD,
    CACHE_TTL_RECENT_MATCHES,
    DB_ELO_PATH,
    ELO_CALIBRATION_MIN_COUNT,
    ELO_INITIAL_RATING,
    ELO_K_FACTOR,
    resolve_db_path,
)
from aragora.ranking.calibration_engine import CalibrationEngine, DomainCalibrationEngine
from aragora.ranking.database import EloDatabase
from aragora.ranking.elo_core import (
    apply_elo_changes,
    calculate_new_elo,
    calculate_pairwise_elo_changes,
    expected_score,
)
from aragora.ranking.leaderboard_engine import LeaderboardEngine
from aragora.ranking.match_recorder import (
    build_match_scores,
    check_duplicate_match,
    compute_calibration_k_multipliers,
    determine_winner,
    generate_match_id,
    normalize_match_params,
    save_match,
)
from aragora.ranking.redteam import RedTeamIntegrator, RedTeamResult, VulnerabilitySummary
from aragora.ranking.relationships import (
    RelationshipMetrics,
    RelationshipStats,
    RelationshipTracker,
)
from aragora.ranking.snapshot import (
    read_snapshot_leaderboard,
    read_snapshot_matches,
    write_snapshot,
)
from aragora.ranking.verification import (
    calculate_verification_elo_change,
    calculate_verification_impact,
    update_rating_from_verification,
)
from aragora.utils.cache import TTLCache

# Re-export for backwards compatibility (moved to sql_helpers)
from aragora.utils.sql_helpers import _escape_like_pattern
from aragora.utils.json_helpers import safe_json_loads

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
    "_escape_like_pattern",
    "get_elo_store",
]


# Singleton EloSystem instance
_elo_store: Optional["EloSystem"] = None


def get_elo_store() -> "EloSystem":
    """Get the global EloSystem singleton instance.

    Returns a singleton EloSystem instance, creating it if necessary.
    Uses the default database path from configuration.

    Returns:
        EloSystem: The global ELO store instance
    """
    global _elo_store
    if _elo_store is None:
        _elo_store = EloSystem()
    return _elo_store


# Use centralized config values (can be overridden via environment variables)
DEFAULT_ELO = ELO_INITIAL_RATING
K_FACTOR = ELO_K_FACTOR
CALIBRATION_MIN_COUNT = ELO_CALIBRATION_MIN_COUNT


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

    @property
    def elo_rating(self) -> float:
        """ELO rating (alias for elo)."""
        return self.elo

    @property
    def total_debates(self) -> int:
        """Total debates (alias for debates_count)."""
        return self.debates_count


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
    _rating_cache: TTLCache[AgentRating] = TTLCache(
        maxsize=200, ttl_seconds=CACHE_TTL_RECENT_MATCHES
    )
    _stats_cache: TTLCache[dict] = TTLCache(maxsize=10, ttl_seconds=CACHE_TTL_LB_STATS)
    _calibration_cache: TTLCache[list] = TTLCache(maxsize=20, ttl_seconds=CACHE_TTL_CALIBRATION_LB)

    def __init__(
        self, db_path: str = DB_ELO_PATH, event_emitter: Optional["EventEmitterProtocol"] = None
    ):
        resolved_path = resolve_db_path(db_path)
        self.db_path = Path(resolved_path)
        self._db = EloDatabase(resolved_path)
        self.event_emitter = event_emitter  # For emitting ELO update events

        # Delegate to extracted modules (lazy initialization)
        self._relationship_tracker: RelationshipTracker | None = None
        self._redteam_integrator: RedTeamIntegrator | None = None

        # Leaderboard engine for read-only analytics
        self._leaderboard_engine = LeaderboardEngine(
            db=self._db,
            leaderboard_cache=self._leaderboard_cache,
            stats_cache=self._stats_cache,
            rating_cache=self._rating_cache,
            rating_factory=self._rating_from_row,
        )

        # Calibration engines for tournament and domain-specific calibration
        self._calibration_engine = CalibrationEngine(db_path=resolved_path, elo_system=self)
        self._domain_calibration_engine = DomainCalibrationEngine(
            db_path=resolved_path, elo_system=self
        )

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

    def register_agent(self, agent_name: str, model: Optional[str] = None) -> AgentRating:
        """Ensure an agent exists in the ratings table (legacy compatibility)."""
        _validate_agent_name(agent_name)
        return self.get_rating(agent_name, use_cache=False)

    def initialize_agent(self, agent_name: str, model: Optional[str] = None) -> AgentRating:
        """Backward-compatible alias for register_agent."""
        return self.register_agent(agent_name, model=model)

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

    def _rating_from_row(self, row: tuple) -> AgentRating:
        """Create AgentRating from a database row (leaderboard query format)."""
        return AgentRating(
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
            placeholders = ",".join("?" * len(agent_names))
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

    def _record_elo_history_batch(self, entries: list[tuple[str, float, str | None]]) -> None:
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
        k_multipliers: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Calculate pairwise ELO changes for all participant combinations.

        Args:
            participants: List of agent names
            scores: Dict of agent -> score
            ratings: Dict of agent -> AgentRating
            confidence_weight: Base confidence weight
            k_multipliers: Optional per-agent K-factor multipliers (from calibration)
        """
        k_multipliers = k_multipliers or {}
        return calculate_pairwise_elo_changes(
            participants, scores, ratings, confidence_weight, K_FACTOR, k_multipliers
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
        return apply_elo_changes(elo_changes, ratings, winner, domain, debate_id, DEFAULT_ELO)

    def _compute_calibration_k_multipliers(
        self,
        participants: list[str],
        calibration_tracker: "Any | None" = None,
    ) -> dict[str, float]:
        """Compute per-agent K-factor multipliers based on calibration quality.

        Delegates to match_recorder.compute_calibration_k_multipliers.
        """
        return compute_calibration_k_multipliers(participants, calibration_tracker)

    @staticmethod
    def _build_match_scores(winner: str, loser: str, is_draw: bool) -> dict[str, float]:
        """Build score dict for a two-player match. Delegates to match_recorder."""
        return build_match_scores(winner, loser, is_draw)

    @staticmethod
    def _generate_match_id(
        participants: list[str], task: str | None = None, domain: str | None = None
    ) -> str:
        """Generate a unique match ID. Delegates to match_recorder."""
        return generate_match_id(participants, task, domain)

    def _normalize_match_params(
        self,
        debate_id: str | None,
        participants: list[str] | str | None,
        scores: dict[str, float] | None,
        winner: str | None,
        loser: str | None,
        draw: bool | None,
        task: str | None,
        domain: str | None,
    ) -> tuple[str, list[str] | None, dict[str, float] | None]:
        """Normalize legacy and modern match signatures. Delegates to match_recorder."""
        return normalize_match_params(
            debate_id, participants, scores, winner, loser, draw, task, domain
        )

    def record_match(
        self,
        debate_id: str | None = None,
        participants: list[str] | str | None = None,
        scores: dict[str, float] | None = None,
        domain: str | None = None,
        confidence_weight: float = 1.0,
        calibration_tracker: Optional[object] = None,
        *,
        winner: Optional[str] = None,
        loser: Optional[str] = None,
        draw: Optional[bool] = None,
        task: Optional[str] = None,
    ) -> dict[str, float]:
        """
        Record a match result and update ELO ratings.

        Args:
            debate_id: Unique debate identifier (auto-generated if omitted)
            participants: List of agent names or legacy "loser" string
            scores: Dict of agent -> score (higher is better)
            domain: Optional domain for domain-specific ELO
            confidence_weight: Weight for ELO change (0-1). Lower values reduce
                               ELO impact for low-confidence debates. Default 1.0.
            calibration_tracker: Optional CalibrationTracker instance. When provided,
                               agents with poor calibration (overconfident/underconfident)
                               receive higher K-factor multipliers, making their ELO
                               more volatile as an incentive to improve calibration.
            winner: Legacy winner name (for compatibility)
            loser: Legacy loser name (for compatibility)
            draw: Legacy draw flag (for compatibility)
            task: Legacy task label (used in auto-generated debate_id)

        Returns:
            Dict of agent -> ELO change
        """
        # Normalize legacy and modern signatures
        debate_id, participants_list, scores = self._normalize_match_params(
            debate_id, participants, scores, winner, loser, draw, task, domain
        )

        if not participants_list or scores is None:
            return {}

        # Clamp confidence_weight to valid range
        confidence_weight = max(0.1, min(1.0, confidence_weight))
        if len(participants_list) < 2:
            return {}

        # Check for duplicate match recording to prevent ELO accumulation bug
        cached_changes = check_duplicate_match(self._db, debate_id)
        if cached_changes is not None:
            return cached_changes

        # Determine winner (highest score)
        winner = determine_winner(scores)

        # Get current ratings (batch query to avoid N+1)
        ratings = self.get_ratings_batch(participants_list)

        # Compute calibration-based K-factor multipliers
        k_multipliers = self._compute_calibration_k_multipliers(
            participants_list, calibration_tracker
        )

        # Calculate pairwise ELO changes (with calibration adjustments if provided)
        elo_changes = self._calculate_pairwise_elo_changes(
            participants_list, scores, ratings, confidence_weight, k_multipliers
        )

        # Apply changes and collect for batch save
        ratings_to_save, history_entries = self._apply_elo_changes(
            elo_changes, ratings, winner, domain, debate_id
        )

        # Batch save all ratings and history (single transaction each)
        self._save_ratings_batch(ratings_to_save)
        self._record_elo_history_batch(history_entries)

        # Save match
        self._save_match(debate_id, winner, participants_list, domain, scores, elo_changes)

        # Write JSON snapshot for fast reads (avoids SQLite locking)
        self._write_snapshot()

        # Invalidate related caches so API returns fresh data
        try:
            from aragora.server.handlers.base import invalidate_on_event

            invalidate_on_event("match_recorded")
        except ImportError:
            pass  # Handlers may not be available in all contexts

        # Emit ELO update events for each agent
        if self.event_emitter and elo_changes:
            try:
                from aragora.server.stream.events import StreamEvent, StreamEventType

                for agent_name, elo_change in elo_changes.items():
                    new_rating = ratings.get(agent_name, 1500) + elo_change
                    self.event_emitter.emit(
                        StreamEvent(
                            type=StreamEventType.AGENT_ELO_UPDATED,
                            data={
                                "agent": agent_name,
                                "elo_change": elo_change,
                                "new_elo": new_rating,
                                "debate_id": debate_id,
                                "domain": domain,
                            },
                        )
                    )
            except (ImportError, AttributeError, TypeError):
                pass  # Stream module not available or emitter misconfigured

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
        """Save match to history. Delegates to match_recorder.save_match."""
        save_match(self._db, debate_id, winner, participants, domain, scores, elo_changes)

    def _record_elo_history(
        self, agent_name: str, elo: float, debate_id: str | None = None
    ) -> None:
        """Record ELO at a point in time."""
        with self._db.connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO elo_history (agent_name, elo, debate_id) VALUES (?, ?, ?)",
                (agent_name, elo, debate_id),
            )
            conn.commit()

    def _write_snapshot(self) -> None:
        """Write JSON snapshot for fast reads. Delegates to snapshot module."""
        snapshot_path = self.db_path.parent / "elo_snapshot.json"
        write_snapshot(snapshot_path, self.get_leaderboard, self.get_recent_matches)

    def get_snapshot_leaderboard(self, limit: int = 20) -> list[dict]:
        """Get leaderboard from JSON snapshot file. Delegates to snapshot module."""
        snapshot_path = self.db_path.parent / "elo_snapshot.json"
        result = read_snapshot_leaderboard(snapshot_path, limit)
        if result is not None:
            return result
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
        """Get recent matches from cache if available. Delegates to snapshot module."""
        snapshot_path = self.db_path.parent / "elo_snapshot.json"
        result = read_snapshot_matches(snapshot_path, limit)
        if result is not None:
            return result
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
        """Get top agents by ELO. Delegates to LeaderboardEngine."""
        return self._leaderboard_engine.get_leaderboard(limit=limit, domain=domain)

    def get_cached_leaderboard(
        self, limit: int = 20, domain: str | None = None
    ) -> list[AgentRating]:
        """Get leaderboard with caching. Delegates to LeaderboardEngine."""
        return self._leaderboard_engine.get_cached_leaderboard(limit=limit, domain=domain)

    def invalidate_leaderboard_cache(self) -> int:
        """Invalidate all cached leaderboard data. Call after rating changes."""
        self._calibration_cache.clear()
        return self._leaderboard_engine.invalidate_leaderboard_cache()

    def invalidate_rating_cache(self, agent_name: str | None = None) -> int:
        """Invalidate cached ratings. Delegates to LeaderboardEngine."""
        return self._leaderboard_engine.invalidate_rating_cache(agent_name)

    def get_top_agents_for_domain(self, domain: str, limit: int = 5) -> list[AgentRating]:
        """Get agents ranked by domain-specific performance. Delegates to LeaderboardEngine."""
        return self._leaderboard_engine.get_top_agents_for_domain(domain=domain, limit=limit)

    def get_elo_history(self, agent_name: str, limit: int = 50) -> list[tuple[str, float]]:
        """Get ELO history for an agent. Delegates to LeaderboardEngine."""
        return self._leaderboard_engine.get_elo_history(agent_name, limit)

    def get_recent_matches(self, limit: int = 10) -> list[dict]:
        """Get recent match results with ELO changes. Delegates to LeaderboardEngine."""
        return self._leaderboard_engine.get_recent_matches(limit)

    def get_head_to_head(self, agent_a: str, agent_b: str) -> dict:
        """Get head-to-head statistics between two agents. Delegates to LeaderboardEngine."""
        return self._leaderboard_engine.get_head_to_head(agent_a, agent_b)

    def get_stats(self, use_cache: bool = True) -> dict:
        """Get overall system statistics. Delegates to LeaderboardEngine."""
        return self._leaderboard_engine.get_stats(use_cache)

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
        """Record an agent's prediction for a tournament winner. Delegates to CalibrationEngine."""
        self._calibration_engine.record_winner_prediction(
            tournament_id, predictor_agent, predicted_winner, confidence
        )

    def resolve_tournament_calibration(
        self,
        tournament_id: str,
        actual_winner: str,
    ) -> dict[str, float]:
        """Resolve tournament and update calibration scores. Delegates to CalibrationEngine."""
        return self._calibration_engine.resolve_tournament(tournament_id, actual_winner)

    def get_calibration_leaderboard(
        self, limit: int = 20, use_cache: bool = True
    ) -> list[AgentRating]:
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

    def get_agent_calibration_history(self, agent_name: str, limit: int = 50) -> list[dict]:
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
        """Convert confidence to bucket key. Delegates to DomainCalibrationEngine."""
        return DomainCalibrationEngine.get_bucket_key(confidence)

    def record_domain_prediction(
        self,
        agent_name: str,
        domain: str,
        confidence: float,
        correct: bool,
    ) -> None:
        """Record a domain-specific prediction. Delegates to DomainCalibrationEngine."""
        self._domain_calibration_engine.record_prediction(agent_name, domain, confidence, correct)

    def get_domain_calibration(self, agent_name: str, domain: Optional[str] = None) -> dict:
        """Get calibration statistics for an agent. Delegates to DomainCalibrationEngine."""
        return self._domain_calibration_engine.get_domain_stats(agent_name, domain)

    def get_calibration_by_bucket(
        self, agent_name: str, domain: Optional[str] = None
    ) -> list[dict]:
        """Get calibration broken down by confidence bucket. Delegates to DomainCalibrationEngine."""
        buckets = self._domain_calibration_engine.get_calibration_curve(agent_name, domain)
        # Convert BucketStats to dict for backwards compatibility
        return [
            {
                "bucket_key": b.bucket_key,
                "bucket_start": b.bucket_start,
                "bucket_end": b.bucket_end,
                "predictions": b.predictions,
                "correct": b.correct,
                "accuracy": b.accuracy,
                "expected_accuracy": b.expected_accuracy,
                "brier_score": b.brier_score,
            }
            for b in buckets
        ]

    def get_expected_calibration_error(self, agent_name: str) -> float:
        """Calculate Expected Calibration Error. Delegates to DomainCalibrationEngine."""
        return self._domain_calibration_engine.get_expected_calibration_error(agent_name)

    def get_best_domains(self, agent_name: str, limit: int = 5) -> list[tuple[str, float]]:
        """Get domains where agent is best calibrated. Delegates to DomainCalibrationEngine."""
        return self._domain_calibration_engine.get_best_domains(agent_name, limit=limit)

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

    # =========================================================================
    # Formal Verification Integration (Phase 10E)
    # Delegates to verification module for cleaner separation of concerns.
    # =========================================================================

    def update_from_verification(
        self,
        agent_name: str,
        domain: str,
        verified_count: int,
        disproven_count: int = 0,
        k_factor: float = 16.0,
    ) -> float:
        """
        Adjust ELO based on formal verification results.

        Delegates to verification module. See verification.py for details.
        """
        _validate_agent_name(agent_name)

        # Early return only if no claims at all (not when they cancel out)
        if verified_count == 0 and disproven_count == 0:
            return 0.0

        net_change = calculate_verification_elo_change(verified_count, disproven_count, k_factor)
        rating = self.get_rating(agent_name, use_cache=False)
        old_elo = rating.elo

        # Apply change (may be zero if claims cancel out)
        if net_change != 0.0:
            update_rating_from_verification(rating, domain, net_change, DEFAULT_ELO)
            self._save_rating(rating)

        # Always record history for verification events (even zero-change ones)
        self._record_elo_history(
            agent_name,
            rating.elo,
            debate_id=f"verification:{domain}:{verified_count}v{disproven_count}d",
        )

        logger.info(
            "verification_elo_update agent=%s domain=%s verified=%d disproven=%d "
            "change=%.1f old_elo=%.1f new_elo=%.1f",
            agent_name,
            domain,
            verified_count,
            disproven_count,
            net_change,
            old_elo,
            rating.elo,
        )
        return net_change

    def get_verification_impact(self, agent_name: str) -> dict:
        """Get summary of verification impact on an agent's ELO. Delegates to verification module."""
        _validate_agent_name(agent_name)
        return calculate_verification_impact(self._db, agent_name)
