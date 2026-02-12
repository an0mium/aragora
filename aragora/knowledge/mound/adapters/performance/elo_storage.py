"""
ELO rating storage and retrieval methods for the PerformanceAdapter.

Handles storage and retrieval of:
- Agent rating snapshots
- Match results
- Calibration predictions
- Agent relationship metrics
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional, Protocol

if TYPE_CHECKING:
    from aragora.ranking.elo import AgentRating, MatchResult

logger = logging.getLogger(__name__)


class StorableRelationshipMetrics(Protocol):
    """
    Protocol for relationship metrics that can be stored.

    This is a structural type that describes the expected interface for
    relationship metrics passed to store_relationship(). It allows for
    duck-typed objects (including MagicMock in tests) that have the required
    attributes.
    """

    agent_a: str
    agent_b: str
    debates_together: int
    a_wins_vs_b: int
    b_wins_vs_a: int
    draws: int


class EloStorageHost(Protocol):
    """Protocol defining what the EloStorageMixin expects from its host class."""

    ELO_PREFIX: str
    MIN_DEBATES_FOR_RELATIONSHIP: int
    _ratings: dict[str, dict[str, Any]]
    _matches: dict[str, dict[str, Any]]
    _calibrations: dict[str, dict[str, Any]]
    _relationships: dict[str, dict[str, Any]]
    _agent_ratings: dict[str, list[str]]
    _agent_matches: dict[str, list[str]]
    _domain_ratings: dict[str, list[str]]


class EloStorageMixin:
    """Mixin providing ELO rating storage and retrieval methods.

    Expects the following attributes on the host class:
    - ELO_PREFIX: str
    - MIN_DEBATES_FOR_RELATIONSHIP: int
    - _ratings: dict[str, dict[str, Any]]
    - _matches: dict[str, dict[str, Any]]
    - _calibrations: dict[str, dict[str, Any]]
    - _relationships: dict[str, dict[str, Any]]
    - _agent_ratings: dict[str, list[str]]
    - _agent_matches: dict[str, list[str]]
    - _domain_ratings: dict[str, list[str]]
    """

    # Type hints for attributes provided by the host class (PerformanceAdapter)
    # These are declared here to satisfy type checkers without requiring Protocol inheritance
    ELO_PREFIX: str
    MIN_DEBATES_FOR_RELATIONSHIP: int
    _ratings: dict[str, dict[str, Any]]
    _matches: dict[str, dict[str, Any]]
    _calibrations: dict[str, dict[str, Any]]
    _relationships: dict[str, dict[str, Any]]
    _agent_ratings: dict[str, list[str]]
    _agent_matches: dict[str, list[str]]
    _domain_ratings: dict[str, list[str]]

    # =========================================================================
    # Rating Storage Methods
    # =========================================================================

    def store_rating(
        self,
        rating: AgentRating,
        debate_id: str | None = None,
        reason: str = "match_update",
    ) -> str:
        """
        Store an agent rating snapshot in the Knowledge Mound.

        This is called after each match to track rating progression.

        Args:
            rating: The AgentRating to store
            debate_id: Optional debate ID that triggered this update
            reason: Reason for the rating update

        Returns:
            The rating ID
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        rating_id = f"{self.ELO_PREFIX}{rating.agent_name}_{timestamp.replace(':', '-')}"

        rating_data = {
            "id": rating_id,
            "agent_name": rating.agent_name,
            "elo": rating.elo,
            "domain_elos": rating.domain_elos,
            "wins": rating.wins,
            "losses": rating.losses,
            "draws": rating.draws,
            "debates_count": rating.debates_count,
            "win_rate": rating.win_rate,
            "games_played": rating.games_played,
            "critiques_accepted": rating.critiques_accepted,
            "critiques_total": rating.critiques_total,
            "critique_acceptance_rate": rating.critique_acceptance_rate,
            "calibration_correct": rating.calibration_correct,
            "calibration_total": rating.calibration_total,
            "calibration_accuracy": rating.calibration_accuracy,
            "debate_id": debate_id,
            "reason": reason,
            "created_at": timestamp,
            "original_updated_at": rating.updated_at,
        }

        self._ratings[rating_id] = rating_data

        # Update indices
        if rating.agent_name not in self._agent_ratings:
            self._agent_ratings[rating.agent_name] = []
        self._agent_ratings[rating.agent_name].append(rating_id)

        # Index by domains
        for domain in rating.domain_elos:
            if domain not in self._domain_ratings:
                self._domain_ratings[domain] = []
            self._domain_ratings[domain].append(rating_id)

        logger.info(f"Stored rating: {rating_id} (elo={rating.elo:.1f})")
        return rating_id

    def store_match(self, match: MatchResult) -> str:
        """
        Store a match result in the Knowledge Mound.

        Args:
            match: The MatchResult to store

        Returns:
            The match ID
        """
        match_id = f"{self.ELO_PREFIX}match_{match.debate_id}"

        match_data = {
            "id": match_id,
            "debate_id": match.debate_id,
            "winner": match.winner,
            "participants": match.participants,
            "domain": match.domain,
            "scores": match.scores,
            "created_at": match.created_at,
        }

        self._matches[match_id] = match_data

        # Update indices
        for participant in match.participants:
            if participant not in self._agent_matches:
                self._agent_matches[participant] = []
            self._agent_matches[participant].append(match_id)

        logger.info(f"Stored match: {match_id} (winner={match.winner})")
        return match_id

    def store_calibration(
        self,
        agent_name: str,
        debate_id: str,
        predicted_winner: str | None,
        predicted_confidence: float,
        actual_winner: str | None,
        was_correct: bool,
        brier_score: float,
    ) -> str:
        """
        Store a calibration prediction in the Knowledge Mound.

        Args:
            agent_name: Agent making the prediction
            debate_id: The debate being predicted
            predicted_winner: Predicted winner (None for draw)
            predicted_confidence: Confidence in prediction (0-1)
            actual_winner: Actual winner (None for draw)
            was_correct: Whether prediction was correct
            brier_score: Brier score for this prediction

        Returns:
            The calibration ID
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        cal_id = f"{self.ELO_PREFIX}cal_{agent_name}_{debate_id}"

        cal_data = {
            "id": cal_id,
            "agent_name": agent_name,
            "debate_id": debate_id,
            "predicted_winner": predicted_winner,
            "predicted_confidence": predicted_confidence,
            "actual_winner": actual_winner,
            "was_correct": was_correct,
            "brier_score": brier_score,
            "created_at": timestamp,
        }

        self._calibrations[cal_id] = cal_data

        logger.info(f"Stored calibration: {cal_id} (correct={was_correct})")
        return cal_id

    def store_relationship(self, metrics: StorableRelationshipMetrics) -> str | None:
        """
        Store relationship metrics between agents.

        Args:
            metrics: An object with relationship metrics attributes. Must have:
                agent_a, agent_b, debates_together, a_wins_vs_b, b_wins_vs_a, draws.
                Optionally: avg_elo_diff, synergy_score.

        Returns:
            The relationship ID if stored, None if below threshold
        """
        if metrics.debates_together < self.MIN_DEBATES_FOR_RELATIONSHIP:
            logger.debug(f"Relationship {metrics.agent_a}-{metrics.agent_b} below debate threshold")
            return None

        rel_id = f"{self.ELO_PREFIX}rel_{metrics.agent_a}_{metrics.agent_b}"

        # Access optional attributes with getattr to support duck-typed objects
        avg_elo_diff: float = getattr(metrics, "avg_elo_diff", 0.0)
        synergy_score: float = getattr(metrics, "synergy_score", 0.5)

        rel_data = {
            "id": rel_id,
            "agent_a": metrics.agent_a,
            "agent_b": metrics.agent_b,
            "debates_together": metrics.debates_together,
            "a_wins_vs_b": metrics.a_wins_vs_b,
            "b_wins_vs_a": metrics.b_wins_vs_a,
            "draws": metrics.draws,
            "avg_elo_diff": avg_elo_diff,
            "synergy_score": synergy_score,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        self._relationships[rel_id] = rel_data

        logger.info(f"Stored relationship: {rel_id}")
        return rel_id

    # =========================================================================
    # Rating Retrieval Methods
    # =========================================================================

    def get_rating(self, rating_id: str) -> dict[str, Any] | None:
        """
        Get a specific rating snapshot by ID.

        Args:
            rating_id: The rating ID (may be prefixed with "el_")

        Returns:
            Rating dict or None
        """
        if not rating_id.startswith(self.ELO_PREFIX):
            rating_id = f"{self.ELO_PREFIX}{rating_id}"
        return self._ratings.get(rating_id)

    def get_match(self, match_id: str) -> dict[str, Any] | None:
        """
        Get a specific match by ID.

        Args:
            match_id: The match ID (may be prefixed with "el_match_")

        Returns:
            Match dict or None
        """
        if not match_id.startswith(self.ELO_PREFIX):
            match_id = f"{self.ELO_PREFIX}match_{match_id}"
        return self._matches.get(match_id)

    def get_relationship(self, agent_a: str, agent_b: str) -> dict[str, Any] | None:
        """
        Get relationship metrics between two agents.

        Args:
            agent_a: First agent
            agent_b: Second agent

        Returns:
            Relationship dict or None
        """
        # Check both orderings
        rel_id_1 = f"{self.ELO_PREFIX}rel_{agent_a}_{agent_b}"
        rel_id_2 = f"{self.ELO_PREFIX}rel_{agent_b}_{agent_a}"

        return self._relationships.get(rel_id_1) or self._relationships.get(rel_id_2)

    def get_agent_skill_history(
        self,
        agent_name: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get skill progression history for an agent (ELO ratings).

        This is the key query for team selection and task assignment.

        Args:
            agent_name: The agent name
            limit: Maximum snapshots to return

        Returns:
            List of rating snapshots ordered by time (newest first)
        """
        rating_ids = self._agent_ratings.get(agent_name, [])
        results = []

        for rating_id in rating_ids:
            rating = self._ratings.get(rating_id)
            if rating:
                results.append(rating)

        # Sort by created_at descending
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return results[:limit]

    def get_domain_expertise(
        self,
        domain: str,
        min_elo: float = 1000.0,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Find agents with expertise in a specific domain (from ELO ratings).

        Returns the most recent rating for each agent in the domain.

        Args:
            domain: The domain to search
            min_elo: Minimum ELO in domain
            limit: Maximum results

        Returns:
            List of rating dicts for agents with domain expertise
        """
        rating_ids = self._domain_ratings.get(domain, [])

        # Get most recent rating per agent
        agent_ratings: dict[str, dict[str, Any]] = {}

        for rating_id in rating_ids:
            rating = self._ratings.get(rating_id)
            if not rating:
                continue

            agent = rating["agent_name"]
            domain_elo = rating.get("domain_elos", {}).get(domain, 0)

            if domain_elo < min_elo:
                continue

            # Keep most recent per agent
            if agent not in agent_ratings:
                agent_ratings[agent] = rating
            else:
                existing_time = agent_ratings[agent].get("created_at", "")
                new_time = rating.get("created_at", "")
                if new_time > existing_time:
                    agent_ratings[agent] = rating

        # Sort by domain ELO descending
        results = list(agent_ratings.values())
        results.sort(
            key=lambda x: x.get("domain_elos", {}).get(domain, 0),
            reverse=True,
        )

        return results[:limit]

    def get_agent_matches(
        self,
        agent_name: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get match history for an agent.

        Args:
            agent_name: The agent name
            limit: Maximum matches to return

        Returns:
            List of match dicts ordered by time (newest first)
        """
        match_ids = self._agent_matches.get(agent_name, [])
        results = []

        for match_id in match_ids:
            match = self._matches.get(match_id)
            if match:
                results.append(match)

        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return results[:limit]

    def get_agent_calibration_history(
        self,
        agent_name: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get calibration prediction history for an agent.

        Args:
            agent_name: The agent name
            limit: Maximum predictions to return

        Returns:
            List of calibration dicts
        """
        results = [
            cal for cal in self._calibrations.values() if cal.get("agent_name") == agent_name
        ]

        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return results[:limit]


__all__ = ["EloStorageMixin", "EloStorageHost", "StorableRelationshipMetrics"]
