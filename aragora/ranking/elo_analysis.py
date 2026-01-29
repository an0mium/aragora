"""
Learning Efficiency and Voting Accuracy Analysis.

Extracted from EloSystem to separate analysis/metrics concerns from
core rating operations. Provides:
- Learning efficiency computation (gain rate, consistency, categorization)
- Voting accuracy tracking and ELO bonuses
- Batch analysis operations
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aragora.config import ELO_INITIAL_RATING

if TYPE_CHECKING:
    from aragora.ranking.elo import EloSystem

logger = logging.getLogger(__name__)

DEFAULT_ELO = ELO_INITIAL_RATING

# Maximum agent name length for validation
MAX_AGENT_NAME_LENGTH = 32


def _validate_agent_name(agent_name: str) -> None:
    """Validate agent name length."""
    if len(agent_name) > MAX_AGENT_NAME_LENGTH:
        raise ValueError(
            f"Agent name exceeds {MAX_AGENT_NAME_LENGTH} characters: {len(agent_name)}"
        )


def _record_learning_bonus(agent: str, category: str) -> None:
    """Record learning bonus metric with lazy import."""
    try:
        from aragora.observability.metrics import record_learning_bonus

        record_learning_bonus(agent, category)
    except ImportError:
        pass


def _record_voting_accuracy(result: str) -> None:
    """Record voting accuracy update metric with lazy import."""
    try:
        from aragora.observability.metrics import record_voting_accuracy_update

        record_voting_accuracy_update(result)
    except ImportError:
        pass


def compute_elo_gain_rate(elo_values: list[float]) -> float:
    """Compute average ELO gain per debate from history.

    Uses linear regression slope as the gain rate.
    Optimized to use single pass + closed-form formulas.

    Args:
        elo_values: List of ELO values over time

    Returns:
        Slope of the linear regression (ELO points per debate)
    """
    if len(elo_values) < 2:
        return 0.0

    n = len(elo_values)

    # For x = 0, 1, 2, ..., n-1, use closed-form formulas:
    # sum(x) = n*(n-1)/2
    # sum(x^2) = n*(n-1)*(2n-1)/6
    sum_x = n * (n - 1) // 2
    sum_x2 = n * (n - 1) * (2 * n - 1) // 6

    # Single pass to compute sum_y and sum_xy simultaneously
    sum_y = 0.0
    sum_xy = 0.0
    for i, y in enumerate(elo_values):
        sum_y += y
        sum_xy += i * y

    # Linear regression: slope m = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return slope


def compute_consistency_score(elo_values: list[float]) -> float:
    """Compute consistency of improvement (0-1 scale).

    Measures how steadily the agent improves vs erratic changes.

    Args:
        elo_values: List of ELO values over time

    Returns:
        Consistency score (0.0 to 1.0)
    """
    if len(elo_values) < 3:
        return 0.0

    # Count positive vs negative changes
    changes = [elo_values[i] - elo_values[i - 1] for i in range(1, len(elo_values))]
    positive_changes = sum(1 for c in changes if c > 0)
    total_changes = len(changes)

    if total_changes == 0:
        return 0.0

    # Consistency is proportion of positive changes
    # Adjusted to give 0.5 for random, higher for consistent improvement
    raw_consistency = positive_changes / total_changes
    return raw_consistency


def categorize_learning(gain_rate: float, consistency: float) -> str:
    """Categorize learning efficiency based on metrics.

    Args:
        gain_rate: ELO gain rate from compute_elo_gain_rate
        consistency: Consistency score from compute_consistency_score

    Returns:
        'rapid': Fast and consistent improvement
        'steady': Slow but consistent improvement
        'slow': Little improvement
        'declining': Getting worse
    """
    if gain_rate > 5 and consistency > 0.6:
        return "rapid"
    elif gain_rate > 2 and consistency > 0.5:
        return "steady"
    elif gain_rate > 0:
        return "slow"
    else:
        return "declining"


def get_learning_efficiency(
    elo_system: "EloSystem",
    agent_name: str,
    domain: str | None = None,
    window_debates: int = 20,
) -> dict:
    """
    Compute learning efficiency for an agent based on ELO improvement rate.

    Learning efficiency measures how quickly an agent improves over time.
    Higher efficiency means the agent learns from debate outcomes and
    improves their performance on similar tasks.

    Args:
        elo_system: EloSystem instance for data access
        agent_name: Name of the agent
        domain: Optional domain filter for domain-specific efficiency
        window_debates: Number of recent debates to analyze

    Returns:
        Dict with learning efficiency metrics:
        - elo_gain_rate: Average ELO gain per debate
        - win_rate_improvement: Change in win rate over window
        - consistency_score: How consistent the improvement is (0-1)
        - learning_category: 'rapid', 'steady', 'slow', or 'declining'
    """
    _validate_agent_name(agent_name)
    rating = elo_system.get_rating(agent_name)

    # Get ELO history for trend analysis
    history = elo_system.get_elo_history(agent_name, limit=window_debates)

    if len(history) < 3:
        return {
            "agent_name": agent_name,
            "domain": domain,
            "elo_gain_rate": 0.0,
            "win_rate_improvement": 0.0,
            "consistency_score": 0.0,
            "learning_category": "insufficient_data",
            "has_meaningful_data": False,
        }

    # Extract ELO values (history is [(timestamp, elo), ...])
    elo_values = [h[1] for h in history]

    # Compute learning metrics
    elo_gain_rate = compute_elo_gain_rate(elo_values)
    consistency = compute_consistency_score(elo_values)
    category = categorize_learning(elo_gain_rate, consistency)

    # For domain-specific, use domain ELO trend
    domain_elo = None
    if domain and rating.domain_elos:
        domain_elo = rating.domain_elos.get(domain)

    # Win rate improvement (compare first half vs second half)
    total_games = rating.wins + rating.losses + rating.draws
    win_rate_improvement = 0.0
    if total_games >= 6:
        # Simple approximation using overall win rate trend
        # In practice, would need per-debate win tracking
        current_win_rate = rating.win_rate
        # Assume starting from baseline of 50%
        win_rate_improvement = current_win_rate - 0.5

    return {
        "agent_name": agent_name,
        "domain": domain,
        "elo_gain_rate": elo_gain_rate,
        "win_rate_improvement": win_rate_improvement,
        "consistency_score": consistency,
        "learning_category": category,
        "current_elo": rating.elo,
        "domain_elo": domain_elo,
        "debates_analyzed": len(history),
        "has_meaningful_data": len(history) >= 5,
    }


def get_learning_efficiency_batch(
    elo_system: "EloSystem",
    agent_names: list[str],
    domain: str | None = None,
    window_debates: int = 20,
) -> dict[str, dict]:
    """Get learning efficiency for multiple agents with batch optimization.

    Args:
        elo_system: EloSystem instance for data access
        agent_names: List of agent names
        domain: Optional domain filter
        window_debates: Number of recent debates to analyze

    Returns:
        Dict mapping agent_name to efficiency metrics dict
    """
    # Batch fetch ratings
    ratings = elo_system.get_ratings_batch(agent_names)
    results = {}

    for name in agent_names:
        rating = ratings.get(name)
        if not rating:
            from aragora.ranking.elo import AgentRating

            rating = AgentRating(agent_name=name)

        # Get ELO history (still individual queries, but ratings are cached)
        history = elo_system.get_elo_history(name, limit=window_debates)

        if len(history) < 3:
            results[name] = {
                "agent_name": name,
                "domain": domain,
                "elo_gain_rate": 0.0,
                "win_rate_improvement": 0.0,
                "consistency_score": 0.0,
                "learning_category": "insufficient_data",
                "has_meaningful_data": False,
            }
            continue

        elo_values = [h[1] for h in history]
        elo_gain_rate = compute_elo_gain_rate(elo_values)
        consistency = compute_consistency_score(elo_values)
        category = categorize_learning(elo_gain_rate, consistency)

        domain_elo = None
        if domain and rating.domain_elos:
            domain_elo = rating.domain_elos.get(domain)

        total_games = rating.wins + rating.losses + rating.draws
        win_rate_improvement = 0.0
        if total_games >= 6:
            win_rate_improvement = rating.win_rate - 0.5

        results[name] = {
            "agent_name": name,
            "domain": domain,
            "elo_gain_rate": elo_gain_rate,
            "win_rate_improvement": win_rate_improvement,
            "consistency_score": consistency,
            "learning_category": category,
            "current_elo": rating.elo,
            "domain_elo": domain_elo,
            "debates_analyzed": len(history),
            "has_meaningful_data": len(history) >= 5,
        }

    return results


def apply_learning_bonus(
    elo_system: "EloSystem",
    agent_name: str,
    domain: str = "general",
    debate_id: str | None = None,
    bonus_factor: float = 0.5,
) -> float:
    """
    Apply ELO bonus based on agent's learning efficiency.

    Rewards agents who demonstrate consistent improvement over time.
    This creates a feedback loop where learning from debates is rewarded.

    Args:
        elo_system: EloSystem instance for data access
        agent_name: Name of the agent
        domain: Debate domain
        debate_id: Optional debate ID for history
        bonus_factor: Multiplier for learning bonus (default 0.5)

    Returns:
        ELO change applied (0 if no bonus)
    """
    _validate_agent_name(agent_name)

    efficiency = get_learning_efficiency(elo_system, agent_name, domain=domain)

    if not efficiency.get("has_meaningful_data"):
        return 0.0

    # Compute bonus based on learning category
    category = efficiency.get("learning_category", "slow")
    gain_rate = efficiency.get("elo_gain_rate", 0.0)
    consistency = efficiency.get("consistency_score", 0.0)

    # Base bonus on learning rate and consistency
    if category == "rapid":
        bonus = bonus_factor * 3.0  # Significant bonus for rapid learners
    elif category == "steady":
        bonus = bonus_factor * 1.5  # Moderate bonus for steady learners
    elif category == "slow":
        bonus = bonus_factor * 0.5  # Small bonus for slow but positive learners
    else:
        return 0.0  # No bonus for declining

    # Scale by consistency
    bonus *= consistency

    if bonus <= 0:
        return 0.0

    # Apply bonus to domain ELO
    rating = elo_system.get_rating(agent_name)
    domain_elos = rating.domain_elos or {}
    old_elo = domain_elos.get(domain, DEFAULT_ELO)
    domain_elos[domain] = old_elo + bonus
    rating.domain_elos = domain_elos
    elo_system._save_rating(rating)

    # Record history
    if debate_id:
        elo_system._record_elo_history(
            agent_name,
            rating.elo,
            debate_id=f"learning:{debate_id}:{category}",
        )

    logger.debug(
        "learning_bonus agent=%s domain=%s category=%s gain_rate=%.2f consistency=%.2f bonus=%.2f",
        agent_name,
        domain,
        category,
        gain_rate,
        consistency,
        bonus,
    )

    _record_learning_bonus(agent_name, category)
    return bonus


def update_voting_accuracy(
    elo_system: "EloSystem",
    agent_name: str,
    voted_for_consensus: bool,
    domain: str = "general",
    debate_id: str | None = None,
    apply_elo_bonus: bool = True,
    bonus_k_factor: float = 4.0,
) -> float:
    """
    Update an agent's voting accuracy and optionally apply ELO bonus.

    Agents who consistently vote for the winning consensus demonstrate
    good judgment and should be rewarded. This creates a feedback loop
    where voting patterns inform agent skill assessment.

    Args:
        elo_system: EloSystem instance for data access
        agent_name: Name of the agent
        voted_for_consensus: Whether the agent voted for the consensus winner
        domain: Debate domain for domain-specific tracking
        debate_id: Optional debate ID for history tracking
        apply_elo_bonus: Whether to apply ELO bonus/penalty
        bonus_k_factor: K-factor for voting bonus (lower = smaller impact)

    Returns:
        ELO change applied (may be 0 if bonuses disabled)
    """
    _validate_agent_name(agent_name)

    rating = elo_system.get_rating(agent_name, use_cache=False)

    # Update calibration-style tracking (reusing existing fields for voting accuracy)
    rating.calibration_total += 1
    if voted_for_consensus:
        rating.calibration_correct += 1

    # Calculate voting accuracy rate
    voting_accuracy = rating.calibration_correct / rating.calibration_total

    elo_change = 0.0
    if apply_elo_bonus:
        # Apply small ELO bonus for consistent correct voting
        # Bonus scales with voting accuracy and sample size
        if rating.calibration_total >= 5:  # Minimum samples for meaningful bonus
            if voted_for_consensus:
                # Bonus for voting with consensus
                # Scale by how much above 50% the agent is
                accuracy_bonus = (voting_accuracy - 0.5) * 2  # 0-1 range
                elo_change = bonus_k_factor * accuracy_bonus
            else:
                # Small penalty for voting against consensus
                # Less severe since dissent can be valuable
                elo_change = -bonus_k_factor * 0.25

            if elo_change != 0:
                # Update domain ELO
                domain_elos = rating.domain_elos or {}
                domain_elo = domain_elos.get(domain, DEFAULT_ELO)
                domain_elos[domain] = domain_elo + elo_change
                rating.domain_elos = domain_elos

    # Save updated rating
    elo_system._save_rating(rating)

    # Record history
    if debate_id:
        elo_system._record_elo_history(
            agent_name,
            rating.elo,
            debate_id=f"voting:{debate_id}:{'correct' if voted_for_consensus else 'incorrect'}",
        )

    logger.debug(
        "voting_accuracy_update agent=%s voted_for_consensus=%s "
        "accuracy=%.2f total=%d elo_change=%.2f",
        agent_name,
        voted_for_consensus,
        voting_accuracy,
        rating.calibration_total,
        elo_change,
    )

    _record_voting_accuracy("correct" if voted_for_consensus else "incorrect")
    return elo_change


def get_voting_accuracy(
    elo_system: "EloSystem",
    agent_name: str,
) -> dict:
    """
    Get voting accuracy statistics for an agent.

    Args:
        elo_system: EloSystem instance for data access
        agent_name: Name of the agent

    Returns:
        Dict with voting accuracy metrics
    """
    _validate_agent_name(agent_name)
    rating = elo_system.get_rating(agent_name)

    total = rating.calibration_total
    correct = rating.calibration_correct

    return {
        "agent_name": agent_name,
        "total_votes": total,
        "correct_votes": correct,
        "accuracy": correct / total if total > 0 else 0.0,
        "has_meaningful_data": total >= 5,
    }


def get_voting_accuracy_batch(
    elo_system: "EloSystem",
    agent_names: list[str],
) -> dict[str, dict]:
    """Get voting accuracy statistics for multiple agents in one query.

    Args:
        elo_system: EloSystem instance for data access
        agent_names: List of agent names

    Returns:
        Dict mapping agent_name to accuracy metrics dict
    """
    ratings = elo_system.get_ratings_batch(agent_names)
    results = {}

    for name, rating in ratings.items():
        total = rating.calibration_total
        correct = rating.calibration_correct
        results[name] = {
            "agent_name": name,
            "total_votes": total,
            "correct_votes": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "has_meaningful_data": total >= 5,
        }

    return results


__all__ = [
    "compute_elo_gain_rate",
    "compute_consistency_score",
    "categorize_learning",
    "get_learning_efficiency",
    "get_learning_efficiency_batch",
    "apply_learning_bonus",
    "update_voting_accuracy",
    "get_voting_accuracy",
    "get_voting_accuracy_batch",
]
