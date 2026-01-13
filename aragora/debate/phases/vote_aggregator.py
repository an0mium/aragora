"""
Vote aggregation for consensus phase.

This module extracts the vote aggregation logic from ConsensusPhase,
providing a clean interface for:
- Grouping similar votes
- Counting weighted votes
- Adding user votes
- Computing vote statistics
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from aragora.core import Vote

logger = logging.getLogger(__name__)


@dataclass
class AggregatedVotes:
    """Result of vote aggregation.

    Contains the aggregated vote counts, totals, and mappings
    needed for consensus determination.
    """

    vote_counts: Counter = field(default_factory=Counter)
    total_weighted: float = 0.0
    choice_mapping: dict[str, str] = field(default_factory=dict)
    vote_groups: dict[str, list[str]] = field(default_factory=dict)

    # Statistics
    total_votes: int = 0
    user_votes_count: int = 0
    agent_votes_count: int = 0

    def get_winner(self) -> tuple[str, float] | None:
        """Get the winning choice and its count.

        Returns:
            Tuple of (winner_choice, vote_count) or None if no votes
        """
        if not self.vote_counts:
            return None
        most_common = self.vote_counts.most_common(1)
        if most_common:
            return most_common[0]
        return None

    def get_confidence(self) -> float:
        """Calculate confidence as winner votes / total votes."""
        winner = self.get_winner()
        if not winner or self.total_weighted <= 0:
            return 0.5  # Neutral default
        return winner[1] / self.total_weighted

    def get_vote_distribution(self) -> dict[str, float]:
        """Get vote distribution as percentages."""
        if self.total_weighted <= 0:
            return {}
        return {choice: count / self.total_weighted for choice, count in self.vote_counts.items()}


class VoteAggregator:
    """Aggregate votes with weighting and grouping.

    Usage:
        aggregator = VoteAggregator(
            group_similar_votes=similarity_fn,
            user_vote_weight=0.5,
        )

        # Aggregate votes
        result = aggregator.aggregate(
            votes=votes,
            weights=weight_cache,
            user_votes=user_votes,
        )

        # Get winner
        winner, count = result.get_winner()
    """

    def __init__(
        self,
        group_similar_votes: Optional[Callable[[list], dict[str, list[str]]]] = None,
        user_vote_weight: float = 0.5,
        user_vote_multiplier: Optional[Callable[[int, Any], float]] = None,
        protocol: Any = None,
    ):
        """Initialize the vote aggregator.

        Args:
            group_similar_votes: Callback to group similar vote choices
            user_vote_weight: Base weight for user votes
            user_vote_multiplier: Callback to compute intensity multiplier
            protocol: Protocol for configuration access
        """
        self._group_similar_votes = group_similar_votes
        self._base_user_weight = user_vote_weight
        self._user_vote_multiplier = user_vote_multiplier
        self.protocol = protocol

    def aggregate(
        self,
        votes: list["Vote"],
        weights: Optional[dict[str, float]] = None,
        user_votes: Optional[list[dict]] = None,
    ) -> AggregatedVotes:
        """Aggregate votes with weighting and grouping.

        Args:
            votes: List of Vote objects from agents
            weights: Dict mapping agent names to vote weights
            user_votes: List of user vote dicts with 'choice' and 'intensity'

        Returns:
            AggregatedVotes with counts and statistics
        """
        weights = weights or {}
        user_votes = user_votes or []

        # Step 1: Group similar votes
        vote_groups, choice_mapping = self._compute_vote_groups(votes)

        # Step 2: Count weighted agent votes
        vote_counts, total_weighted = self._count_weighted_votes(votes, choice_mapping, weights)
        agent_votes_count = len([v for v in votes if not isinstance(v, Exception)])

        # Step 3: Add user votes
        vote_counts, total_weighted, user_count = self._add_user_votes(
            vote_counts, total_weighted, choice_mapping, user_votes
        )

        return AggregatedVotes(
            vote_counts=vote_counts,
            total_weighted=total_weighted,
            choice_mapping=choice_mapping,
            vote_groups=vote_groups,
            total_votes=agent_votes_count + user_count,
            agent_votes_count=agent_votes_count,
            user_votes_count=user_count,
        )

    def _compute_vote_groups(
        self, votes: list["Vote"]
    ) -> tuple[dict[str, list[str]], dict[str, str]]:
        """Group similar votes and create choice mapping.

        Args:
            votes: List of votes to group

        Returns:
            Tuple of (vote_groups, choice_mapping)
        """
        if not self._group_similar_votes:
            # No grouping - identity mapping
            choices = set(
                v.choice for v in votes if not isinstance(v, Exception) and hasattr(v, "choice")
            )
            return {c: [c] for c in choices}, {c: c for c in choices}

        try:
            vote_groups = self._group_similar_votes(votes)
        except Exception as e:
            logger.warning(f"Vote grouping failed: {e}")
            choices = set(
                v.choice for v in votes if not isinstance(v, Exception) and hasattr(v, "choice")
            )
            return {c: [c] for c in choices}, {c: c for c in choices}

        # Build choice mapping from groups
        choice_mapping: dict[str, str] = {}
        for canonical, variants in vote_groups.items():
            for variant in variants:
                choice_mapping[variant] = canonical

        if vote_groups:
            logger.debug(f"vote_grouping_merged groups={vote_groups}")

        return vote_groups, choice_mapping

    def _count_weighted_votes(
        self,
        votes: list["Vote"],
        choice_mapping: dict[str, str],
        weights: dict[str, float],
    ) -> tuple[Counter, float]:
        """Count weighted votes from agents.

        Args:
            votes: List of Vote objects
            choice_mapping: Mapping from raw choice to canonical choice
            weights: Mapping from agent name to vote weight

        Returns:
            Tuple of (vote_counts, total_weighted)
        """
        vote_counts: Counter = Counter()
        total_weighted = 0.0

        for vote in votes:
            if isinstance(vote, Exception):
                continue
            if not hasattr(vote, "choice") or not hasattr(vote, "agent"):
                continue

            canonical = choice_mapping.get(vote.choice, vote.choice)
            weight = weights.get(vote.agent, 1.0)

            vote_counts[canonical] += weight  # type: ignore[assignment]
            total_weighted += weight

        return vote_counts, total_weighted

    def _add_user_votes(
        self,
        vote_counts: Counter,
        total_weighted: float,
        choice_mapping: dict[str, str],
        user_votes: list[dict],
    ) -> tuple[Counter, float, int]:
        """Add user votes to counts.

        Args:
            vote_counts: Existing vote counts
            total_weighted: Existing total weight
            choice_mapping: Choice mapping for normalization
            user_votes: List of user vote dicts

        Returns:
            Tuple of (updated_counts, updated_total, user_vote_count)
        """
        user_count = 0

        for user_vote in user_votes:
            choice = user_vote.get("choice", "")
            if not choice:
                continue

            canonical = choice_mapping.get(choice, choice)
            intensity = user_vote.get("intensity", 5)

            # Calculate weight
            if self._user_vote_multiplier:
                try:
                    intensity_multiplier = self._user_vote_multiplier(intensity, self.protocol)
                except Exception as e:
                    logger.warning("User vote multiplier failed, using 1.0: %s", e)
                    intensity_multiplier = 1.0
            else:
                intensity_multiplier = 1.0

            final_weight = self._base_user_weight * intensity_multiplier
            vote_counts[canonical] += final_weight  # type: ignore[assignment]
            total_weighted += final_weight
            user_count += 1

            logger.debug(
                f"user_vote user={user_vote.get('user_id', 'anonymous')} "
                f"choice={choice} intensity={intensity} weight={final_weight:.2f}"
            )

        return vote_counts, total_weighted, user_count

    def count_unweighted(
        self,
        votes: list["Vote"],
        choice_mapping: Optional[dict[str, str]] = None,
    ) -> Counter:
        """Count votes without weighting (for unanimous mode).

        Args:
            votes: List of Vote objects
            choice_mapping: Optional mapping for choice normalization

        Returns:
            Counter of vote counts by choice
        """
        choice_mapping = choice_mapping or {}
        vote_counts: Counter = Counter()

        for vote in votes:
            if isinstance(vote, Exception):
                continue
            if not hasattr(vote, "choice"):
                continue

            canonical = choice_mapping.get(vote.choice, vote.choice)
            vote_counts[canonical] += 1

        return vote_counts


def calculate_consensus_strength(
    vote_counts: Counter,
) -> tuple[str, float]:
    """Calculate consensus strength from vote distribution.

    Args:
        vote_counts: Counter of votes by choice

    Returns:
        Tuple of (strength_label, variance)
        - "unanimous" if only one choice
        - "strong" if variance < 1
        - "medium" if variance < 2
        - "weak" otherwise
    """
    if len(vote_counts) <= 1:
        return "unanimous", 0.0

    counts = list(vote_counts.values())
    mean = sum(counts) / len(counts)
    variance = sum((c - mean) ** 2 for c in counts) / len(counts)

    if variance < 1:
        strength = "strong"
    elif variance < 2:
        strength = "medium"
    else:
        strength = "weak"

    return strength, variance
