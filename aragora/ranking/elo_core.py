"""
Core ELO calculation functions.

Pure functions for ELO rating calculations, extracted from EloSystem
to improve testability and reduce module complexity.

These functions implement the standard ELO rating system used in chess
and adapted for multi-agent debate ranking.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from aragora.config import ELO_INITIAL_RATING, ELO_K_FACTOR

if TYPE_CHECKING:
    from aragora.ranking.elo import AgentRating

# Re-export config values for convenience
DEFAULT_ELO = ELO_INITIAL_RATING
K_FACTOR = ELO_K_FACTOR


def expected_score(elo_a: float, elo_b: float) -> float:
    """
    Calculate expected score for player A against player B.

    Based on the standard ELO formula:
    E_A = 1 / (1 + 10^((R_B - R_A) / 400))

    Args:
        elo_a: ELO rating of player A
        elo_b: ELO rating of player B

    Returns:
        Expected score for player A (0.0 to 1.0)
    """
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def calculate_new_elo(
    current_elo: float,
    expected: float,
    actual: float,
    k: float = K_FACTOR,
) -> float:
    """
    Calculate new ELO rating after a match.

    Uses the standard ELO formula:
    R'_A = R_A + K * (S_A - E_A)

    Args:
        current_elo: Current ELO rating
        expected: Expected score (from expected_score function)
        actual: Actual score (0=loss, 0.5=draw, 1=win)
        k: K-factor (volatility of rating changes)

    Returns:
        New ELO rating
    """
    return current_elo + k * (actual - expected)


def calculate_pairwise_elo_changes(
    participants: list[str],
    scores: dict[str, float],
    ratings: dict[str, "AgentRating"],
    confidence_weight: float = 1.0,
    k_factor: float = K_FACTOR,
    k_multipliers: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Calculate pairwise ELO changes for all participant combinations.

    In a multi-agent debate, each pair of agents is compared and
    ELO changes are accumulated. This ensures the rating changes
    reflect performance against all opponents.

    Args:
        participants: List of agent names
        scores: Dict of agent -> score in the debate
        ratings: Dict of agent -> current AgentRating
        confidence_weight: Weight for ELO change (0-1), lower for uncertain outcomes
        k_factor: K-factor for rating volatility
        k_multipliers: Optional per-agent K-factor multipliers from calibration.
            Overconfident agents get higher multipliers (1.2-1.5x) to lose more on losses.
            Underconfident agents get lower multipliers (0.7-0.9x) to gain more on wins.
            Well-calibrated agents get 1.0x (no adjustment).

    Returns:
        Dict of agent -> total ELO change
    """
    elo_changes: dict[str, float] = {}
    k_multipliers = k_multipliers or {}
    base_effective_k = k_factor * confidence_weight

    for i, agent_a in enumerate(participants):
        for agent_b in participants[i + 1 :]:
            rating_a = ratings[agent_a]
            rating_b = ratings[agent_b]

            # Expected scores
            expected_a = expected_score(rating_a.elo, rating_b.elo)

            # Actual scores (normalized to 0-1)
            score_a = scores.get(agent_a, 0)
            score_b = scores.get(agent_b, 0)
            total = score_a + score_b
            if total > 0:
                actual_a = score_a / total
            else:
                actual_a = 0.5  # Draw if no scores

            # Apply per-agent K-factor multipliers from calibration
            # Each agent's change is scaled by their own calibration multiplier
            k_mult_a = k_multipliers.get(agent_a, 1.0)
            k_mult_b = k_multipliers.get(agent_b, 1.0)

            # Calculate base ELO change
            base_change = base_effective_k * (actual_a - expected_a)

            # Apply calibration-based multipliers per agent
            # This breaks zero-sum slightly but rewards well-calibrated agents
            change_a = base_change * k_mult_a
            change_b = -base_change * k_mult_b

            elo_changes[agent_a] = elo_changes.get(agent_a, 0) + change_a
            elo_changes[agent_b] = elo_changes.get(agent_b, 0) + change_b

    return elo_changes


def apply_elo_changes(
    elo_changes: dict[str, float],
    ratings: dict[str, "AgentRating"],
    winner: Optional[str],
    domain: Optional[str] = None,
    debate_id: Optional[str] = None,
    default_elo: float = DEFAULT_ELO,
) -> tuple[list["AgentRating"], list[tuple[str, float, Optional[str]]]]:
    """
    Apply ELO changes to ratings and prepare for batch save.

    Updates the rating objects in-place and returns them for persistence.
    Also generates history entries for ELO progression tracking.

    Args:
        elo_changes: Dict of agent -> ELO change
        ratings: Dict of agent -> current AgentRating (modified in place)
        winner: Name of winning agent (None for draw)
        domain: Optional domain for domain-specific ELO
        debate_id: Debate identifier for history tracking
        default_elo: Default ELO for new domain ratings

    Returns:
        Tuple of (ratings_to_save, history_entries)
        - ratings_to_save: List of modified AgentRating objects
        - history_entries: List of (agent_name, new_elo, debate_id) tuples
    """
    ratings_to_save = []
    history_entries: list[tuple[str, float, Optional[str]]] = []
    now = datetime.now().isoformat()

    for agent_name, change in elo_changes.items():
        rating = ratings[agent_name]
        rating.elo += change
        rating.debates_count += 1
        rating.updated_at = now

        # Update win/loss/draw counters
        if winner == agent_name:
            rating.wins += 1
        elif winner is None:
            rating.draws += 1
        else:
            rating.losses += 1

        # Update domain-specific ELO if applicable
        if domain:
            current_domain_elo = rating.domain_elos.get(domain, default_elo)
            rating.domain_elos[domain] = current_domain_elo + change

        ratings_to_save.append(rating)
        history_entries.append((agent_name, rating.elo, debate_id))

    return ratings_to_save, history_entries


def calculate_win_probability(elo_a: float, elo_b: float) -> float:
    """
    Calculate probability that player A beats player B.

    This is equivalent to expected_score but with clearer semantics
    for prediction contexts.

    Args:
        elo_a: ELO rating of player A
        elo_b: ELO rating of player B

    Returns:
        Probability of A winning (0.0 to 1.0)
    """
    return expected_score(elo_a, elo_b)


def elo_diff_for_probability(target_probability: float) -> float:
    """
    Calculate the ELO difference needed for a given win probability.

    Inverse of expected_score function.

    Args:
        target_probability: Desired win probability (0.0 to 1.0)

    Returns:
        ELO difference (positive means higher rated)

    Raises:
        ValueError: If probability is not in (0, 1)
    """
    if target_probability <= 0 or target_probability >= 1:
        raise ValueError("Probability must be in (0, 1)")

    import math

    return 400 * math.log10(target_probability / (1 - target_probability))


__all__ = [
    "DEFAULT_ELO",
    "K_FACTOR",
    "expected_score",
    "calculate_new_elo",
    "calculate_pairwise_elo_changes",
    "apply_elo_changes",
    "calculate_win_probability",
    "elo_diff_for_probability",
]
