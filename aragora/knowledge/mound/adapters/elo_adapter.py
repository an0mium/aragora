"""
EloAdapter - Bridges ELO/Ranking system to the Knowledge Mound.

This adapter enables bidirectional integration between the ELO ranking
system and the Knowledge Mound:

- Data flow IN: Agent ratings, match results, calibration data stored in KM
- Data flow OUT: Agent skill history and domain expertise retrieved
- Reverse flow: KM patterns inform ELO adjustments based on debate quality

The adapter provides:
- Rating storage after matches
- Calibration prediction persistence
- Relationship metrics tracking
- Skill history retrieval for team selection
- **KM pattern → ELO adjustment (reverse flow)**

ID Prefix: el_
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.ranking.elo import AgentRating, EloSystem, MatchResult
    from aragora.ranking.relationships import RelationshipMetrics

logger = logging.getLogger(__name__)


@dataclass
class KMEloPattern:
    """Pattern detected in Knowledge Mound that can influence ELO.

    These patterns are derived from analyzing debate outcomes, claim
    validation, and knowledge contribution across debates.
    """

    agent_name: str
    pattern_type: str  # "success_contributor", "contradiction_source", "domain_expert", "crux_resolver"
    confidence: float  # 0.0-1.0 KM confidence in this pattern
    observation_count: int = 1  # How many times observed
    domain: Optional[str] = None  # Domain if pattern is domain-specific
    debate_ids: List[str] = field(default_factory=list)  # Debates that formed this pattern
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EloAdjustmentRecommendation:
    """Recommendation for ELO adjustment based on KM patterns.

    These are recommendations that should be reviewed before applying,
    as they represent inferred quality from knowledge patterns rather
    than direct match outcomes.
    """

    agent_name: str
    adjustment: float  # Positive = boost, negative = penalty
    reason: str  # Human-readable explanation
    patterns: List[KMEloPattern] = field(default_factory=list)  # Supporting patterns
    confidence: float = 0.7  # Overall confidence in recommendation
    domain: Optional[str] = None  # If domain-specific
    applied: bool = False


@dataclass
class EloSyncResult:
    """Result of syncing KM patterns to ELO."""

    total_patterns: int = 0
    adjustments_recommended: int = 0
    adjustments_applied: int = 0
    adjustments_skipped: int = 0
    total_elo_change: float = 0.0
    agents_affected: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_ms: int = 0


@dataclass
class RatingSearchResult:
    """Wrapper for rating search results with adapter metadata."""

    rating: Dict[str, Any]
    relevance_score: float = 0.0

    def __post_init__(self) -> None:
        pass


class EloAdapter:
    """
    Adapter that bridges EloSystem to the Knowledge Mound.

    Provides methods for the Knowledge Mound's federated query system:
    - store_rating: Store agent rating after match
    - store_match: Store match result for history
    - store_calibration: Store calibration predictions
    - store_relationship: Store agent relationship metrics
    - get_agent_skill_history: Retrieve skill progression
    - get_domain_expertise: Find agents with domain expertise

    Usage:
        from aragora.ranking.elo import EloSystem
        from aragora.knowledge.mound.adapters import EloAdapter

        elo = EloSystem()
        adapter = EloAdapter(elo)

        # After a match, store the updated ratings
        adapter.store_match(match_result)

        # Query skill history for team selection
        history = adapter.get_agent_skill_history("claude")
    """

    ID_PREFIX = "el_"

    # Thresholds
    MIN_DEBATES_FOR_RELATIONSHIP = 5  # Min debates to store relationship metrics

    def __init__(
        self,
        elo_system: Optional["EloSystem"] = None,
        enable_dual_write: bool = False,
    ):
        """
        Initialize the adapter.

        Args:
            elo_system: Optional EloSystem instance to wrap
            enable_dual_write: If True, writes go to both systems during migration
        """
        self._elo_system = elo_system
        self._enable_dual_write = enable_dual_write

        # In-memory storage for queries (will be replaced by KM backend)
        self._ratings: Dict[str, Dict[str, Any]] = {}
        self._matches: Dict[str, Dict[str, Any]] = {}
        self._calibrations: Dict[str, Dict[str, Any]] = {}
        self._relationships: Dict[str, Dict[str, Any]] = {}

        # Indices for fast lookup
        self._agent_ratings: Dict[str, List[str]] = {}  # agent -> [rating_ids] (history)
        self._agent_matches: Dict[str, List[str]] = {}  # agent -> [match_ids]
        self._domain_ratings: Dict[str, List[str]] = {}  # domain -> [rating_ids]

    @property
    def elo_system(self) -> Optional["EloSystem"]:
        """Access the underlying EloSystem."""
        return self._elo_system

    def set_elo_system(self, elo_system: "EloSystem") -> None:
        """Set the ELO system to use."""
        self._elo_system = elo_system

    def store_rating(
        self,
        rating: "AgentRating",
        debate_id: Optional[str] = None,
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
        timestamp = datetime.utcnow().isoformat()
        rating_id = f"{self.ID_PREFIX}{rating.agent_name}_{timestamp.replace(':', '-')}"

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

    def store_match(
        self,
        match: "MatchResult",
    ) -> str:
        """
        Store a match result in the Knowledge Mound.

        Args:
            match: The MatchResult to store

        Returns:
            The match ID
        """
        match_id = f"{self.ID_PREFIX}match_{match.debate_id}"

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
        predicted_winner: Optional[str],
        predicted_confidence: float,
        actual_winner: Optional[str],
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
        timestamp = datetime.utcnow().isoformat()
        cal_id = f"{self.ID_PREFIX}cal_{agent_name}_{debate_id}"

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

    def store_relationship(
        self,
        metrics: "RelationshipMetrics",
    ) -> Optional[str]:
        """
        Store relationship metrics between agents.

        Args:
            metrics: The RelationshipMetrics to store

        Returns:
            The relationship ID if stored, None if below threshold
        """
        if metrics.debates_together < self.MIN_DEBATES_FOR_RELATIONSHIP:
            logger.debug(
                f"Relationship {metrics.agent_a}-{metrics.agent_b} below debate threshold"
            )
            return None

        rel_id = f"{self.ID_PREFIX}rel_{metrics.agent_a}_{metrics.agent_b}"

        rel_data = {
            "id": rel_id,
            "agent_a": metrics.agent_a,
            "agent_b": metrics.agent_b,
            "debates_together": metrics.debates_together,
            "a_wins_vs_b": metrics.a_wins_vs_b,
            "b_wins_vs_a": metrics.b_wins_vs_a,
            "draws": metrics.draws,
            "avg_elo_diff": metrics.avg_elo_diff if hasattr(metrics, 'avg_elo_diff') else 0.0,
            "synergy_score": metrics.synergy_score if hasattr(metrics, 'synergy_score') else 0.5,
            "created_at": datetime.utcnow().isoformat(),
        }

        self._relationships[rel_id] = rel_data

        logger.info(f"Stored relationship: {rel_id}")
        return rel_id

    def get_rating(self, rating_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific rating snapshot by ID.

        Args:
            rating_id: The rating ID (may be prefixed with "el_")

        Returns:
            Rating dict or None
        """
        if not rating_id.startswith(self.ID_PREFIX):
            rating_id = f"{self.ID_PREFIX}{rating_id}"
        return self._ratings.get(rating_id)

    def get_match(self, match_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific match by ID.

        Args:
            match_id: The match ID (may be prefixed with "el_match_")

        Returns:
            Match dict or None
        """
        if not match_id.startswith(self.ID_PREFIX):
            match_id = f"{self.ID_PREFIX}match_{match_id}"
        return self._matches.get(match_id)

    def get_agent_skill_history(
        self,
        agent_name: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get skill progression history for an agent.

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
    ) -> List[Dict[str, Any]]:
        """
        Find agents with expertise in a specific domain.

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
        agent_ratings: Dict[str, Dict[str, Any]] = {}

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
    ) -> List[Dict[str, Any]]:
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
    ) -> List[Dict[str, Any]]:
        """
        Get calibration prediction history for an agent.

        Args:
            agent_name: The agent name
            limit: Maximum predictions to return

        Returns:
            List of calibration dicts
        """
        results = [
            cal for cal in self._calibrations.values()
            if cal.get("agent_name") == agent_name
        ]

        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return results[:limit]

    def get_relationship(
        self,
        agent_a: str,
        agent_b: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get relationship metrics between two agents.

        Args:
            agent_a: First agent
            agent_b: Second agent

        Returns:
            Relationship dict or None
        """
        # Check both orderings
        rel_id_1 = f"{self.ID_PREFIX}rel_{agent_a}_{agent_b}"
        rel_id_2 = f"{self.ID_PREFIX}rel_{agent_b}_{agent_a}"

        return self._relationships.get(rel_id_1) or self._relationships.get(rel_id_2)

    def to_knowledge_item(self, rating: Dict[str, Any]) -> "KnowledgeItem":
        """
        Convert a rating dict to a KnowledgeItem.

        Args:
            rating: The rating dictionary

        Returns:
            KnowledgeItem for unified knowledge mound API
        """
        from aragora.knowledge.unified.types import (
            ConfidenceLevel,
            KnowledgeItem,
            KnowledgeSource,
        )

        # Confidence based on games played (more games = more confident rating)
        games = rating.get("games_played", 0)
        if games >= 50:
            confidence = ConfidenceLevel.VERIFIED
        elif games >= 20:
            confidence = ConfidenceLevel.HIGH
        elif games >= 10:
            confidence = ConfidenceLevel.MEDIUM
        elif games >= 3:
            confidence = ConfidenceLevel.LOW
        else:
            confidence = ConfidenceLevel.UNVERIFIED

        created_at = rating.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                created_at = datetime.utcnow()
        elif created_at is None:
            created_at = datetime.utcnow()

        elo = rating.get("elo", 1000)
        # Normalize ELO to 0-1 importance (1000 = 0.5, 1500 = 0.75, 2000 = 1.0)
        importance = min(1.0, max(0.0, (elo - 500) / 1500))

        content = (
            f"Agent {rating.get('agent_name', 'unknown')}: "
            f"ELO {elo:.0f}, "
            f"{rating.get('wins', 0)}W/{rating.get('losses', 0)}L/{rating.get('draws', 0)}D"
        )

        return KnowledgeItem(
            id=rating["id"],
            content=content,
            source=KnowledgeSource.ELO,
            source_id=rating.get("agent_name", rating["id"]),
            confidence=confidence,
            created_at=created_at,
            updated_at=created_at,
            metadata={
                "agent_name": rating.get("agent_name", ""),
                "elo": elo,
                "win_rate": rating.get("win_rate", 0.0),
                "games_played": games,
                "domain_elos": rating.get("domain_elos", {}),
                "calibration_accuracy": rating.get("calibration_accuracy", 0.0),
                "reason": rating.get("reason", ""),
            },
            importance=importance,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored ELO data."""
        return {
            "total_ratings": len(self._ratings),
            "total_matches": len(self._matches),
            "total_calibrations": len(self._calibrations),
            "total_relationships": len(self._relationships),
            "agents_tracked": len(self._agent_ratings),
            "domains_tracked": len(self._domain_ratings),
        }


__all__ = [
    "EloAdapter",
    "RatingSearchResult",
]
