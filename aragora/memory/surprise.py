"""
Unified Surprise Scoring Module.

Consolidates surprise-based memorization logic used across the memory system.
Based on Titans/MIRAS principles - patterns that deviate from expected outcomes
get higher priority for retention.

Key concepts:
- Surprise = |actual - expected|, measuring prediction error
- High surprise → pattern is novel/unexpected → prioritize learning
- Low surprise → pattern is predictable → safe to demote/forget

Usage:
    from aragora.memory.surprise import (
        calculate_surprise,
        calculate_base_rate,
        update_surprise_ema,
        SurpriseScorer,
    )

    # Simple surprise calculation
    surprise = calculate_surprise(actual=1.0, expected=0.3)  # 0.7

    # With exponential moving average
    new_surprise = update_surprise_ema(old_surprise=0.5, new_surprise=0.8)  # ~0.59

    # Full scorer with state
    scorer = SurpriseScorer()
    score = scorer.score_outcome(category="type_errors", is_success=True)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


# Default EMA alpha for surprise smoothing
DEFAULT_SURPRISE_ALPHA = 0.3


def calculate_surprise(
    actual: float,
    expected: float,
    scale_factor: float = 1.0,
    max_surprise: float = 1.0,
) -> float:
    """
    Calculate surprise score as deviation from expectation.

    Args:
        actual: Actual outcome (typically 0.0 for failure, 1.0 for success)
        expected: Expected outcome (base rate, probability estimate)
        scale_factor: Multiplier for the raw surprise (default 1.0)
        max_surprise: Maximum surprise value to return (default 1.0)

    Returns:
        Surprise score in [0, max_surprise]

    Example:
        >>> calculate_surprise(actual=1.0, expected=0.3)
        0.7
        >>> calculate_surprise(actual=0.0, expected=0.9)
        0.9
    """
    raw_surprise = abs(actual - expected)
    scaled = raw_surprise * scale_factor
    return min(max_surprise, scaled)


def calculate_base_rate(
    success_count: int,
    failure_count: int,
    prior: float = 0.5,
    prior_weight: int = 2,
) -> float:
    """
    Calculate base success rate with Bayesian smoothing.

    Uses a Beta prior to avoid extreme rates with few observations.

    Args:
        success_count: Number of successes observed
        failure_count: Number of failures observed
        prior: Prior probability (default 0.5 = uninformative)
        prior_weight: Effective sample size of prior (default 2)

    Returns:
        Smoothed success rate in [0, 1]

    Example:
        >>> calculate_base_rate(success_count=8, failure_count=2)
        0.75
        >>> calculate_base_rate(success_count=0, failure_count=0)
        0.5  # Falls back to prior with no observations
    """
    total = success_count + failure_count
    if total == 0:
        return prior

    # Bayesian smoothing: (successes + prior*weight) / (total + weight)
    smoothed_rate = (success_count + prior * prior_weight) / (total + prior_weight)
    return smoothed_rate


def update_surprise_ema(
    old_surprise: float,
    new_surprise: float,
    alpha: float = DEFAULT_SURPRISE_ALPHA,
) -> float:
    """
    Update surprise score using exponential moving average.

    EMA provides temporal smoothing to avoid overreacting to single events.

    Args:
        old_surprise: Previous surprise score
        new_surprise: Current surprise observation
        alpha: Smoothing factor in (0, 1). Higher = more weight on new value.

    Returns:
        Updated surprise score

    Example:
        >>> update_surprise_ema(old_surprise=0.5, new_surprise=0.8)
        0.59
        >>> update_surprise_ema(old_surprise=0.5, new_surprise=0.8, alpha=0.5)
        0.65
    """
    return old_surprise * (1 - alpha) + new_surprise * alpha


def calculate_combined_surprise(
    success_surprise: float,
    agent_prediction_error: Optional[float] = None,
    success_weight: float = 0.7,
    agent_weight: float = 0.3,
) -> float:
    """
    Combine multiple surprise signals into a single score.

    Useful when you have both outcome surprise and agent calibration error.

    Args:
        success_surprise: Surprise from success/failure rate deviation
        agent_prediction_error: Optional error from agent confidence calibration
        success_weight: Weight for success surprise (default 0.7)
        agent_weight: Weight for agent prediction error (default 0.3)

    Returns:
        Combined surprise score in [0, 1]
    """
    if agent_prediction_error is None:
        return success_surprise

    combined = success_weight * success_surprise + agent_weight * agent_prediction_error
    return min(1.0, combined)


@dataclass
class CategoryStats:
    """Statistics for a category used in surprise calculation."""

    success_count: int = 0
    failure_count: int = 0
    current_surprise: float = 0.0

    @property
    def total(self) -> int:
        """Total observations."""
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        """Raw success rate (no smoothing)."""
        if self.total == 0:
            return 0.5
        return self.success_count / self.total


@dataclass
class SurpriseScorer:
    """
    Stateful surprise scorer that tracks category statistics.

    Maintains per-category base rates and surprise scores, providing
    a unified interface for surprise-based learning.

    Example:
        scorer = SurpriseScorer()

        # Score outcomes
        s1 = scorer.score_outcome("type_errors", is_success=True)
        s2 = scorer.score_outcome("type_errors", is_success=False)

        # Get category stats
        stats = scorer.get_category_stats("type_errors")
        print(f"Success rate: {stats.success_rate:.2%}")
    """

    alpha: float = DEFAULT_SURPRISE_ALPHA
    scale_factor: float = 1.0
    prior: float = 0.5
    prior_weight: int = 2

    _categories: Dict[str, CategoryStats] = field(default_factory=dict)

    def score_outcome(
        self,
        category: str,
        is_success: bool,
        agent_prediction_error: Optional[float] = None,
    ) -> float:
        """
        Score an outcome and update category statistics.

        Args:
            category: Category/type of the pattern (e.g., "type_errors", "logic_bugs")
            is_success: Whether this was a successful outcome
            agent_prediction_error: Optional agent calibration error

        Returns:
            Updated surprise score for this category
        """
        # Get or create category stats
        if category not in self._categories:
            self._categories[category] = CategoryStats()

        stats = self._categories[category]

        # Calculate base rate before updating counts
        base_rate = calculate_base_rate(
            stats.success_count,
            stats.failure_count,
            prior=self.prior,
            prior_weight=self.prior_weight,
        )

        # Calculate surprise for this outcome
        actual = 1.0 if is_success else 0.0
        new_surprise = calculate_surprise(
            actual=actual,
            expected=base_rate,
            scale_factor=self.scale_factor,
        )

        # Combine with agent prediction error if provided
        if agent_prediction_error is not None:
            new_surprise = calculate_combined_surprise(
                success_surprise=new_surprise,
                agent_prediction_error=agent_prediction_error,
            )

        # Update stats
        if is_success:
            stats.success_count += 1
        else:
            stats.failure_count += 1

        # Update surprise with EMA
        stats.current_surprise = update_surprise_ema(
            old_surprise=stats.current_surprise,
            new_surprise=new_surprise,
            alpha=self.alpha,
        )

        return stats.current_surprise

    def get_category_stats(self, category: str) -> Optional[CategoryStats]:
        """Get statistics for a category."""
        return self._categories.get(category)

    def get_all_categories(self) -> Dict[str, CategoryStats]:
        """Get all category statistics."""
        return self._categories.copy()

    def get_category_surprise(self, category: str) -> float:
        """Get current surprise score for a category."""
        stats = self._categories.get(category)
        return stats.current_surprise if stats else 0.0

    def reset_category(self, category: str) -> None:
        """Reset statistics for a category."""
        if category in self._categories:
            del self._categories[category]

    def reset_all(self) -> None:
        """Reset all category statistics."""
        self._categories.clear()


# Database-aware surprise calculation helpers


def calculate_surprise_from_db_row(
    success_count: int,
    failure_count: int,
    is_success: bool,
    old_surprise: float = 0.0,
    alpha: float = DEFAULT_SURPRISE_ALPHA,
    scale_factor: float = 1.0,
) -> float:
    """
    Calculate updated surprise score from database statistics.

    Convenience function for updating surprise when you have counts from a DB query.

    Args:
        success_count: Number of successes in database
        failure_count: Number of failures in database
        is_success: Whether current outcome is a success
        old_surprise: Previous surprise score
        alpha: EMA smoothing factor
        scale_factor: Surprise scale factor

    Returns:
        Updated surprise score
    """
    base_rate = calculate_base_rate(success_count, failure_count)
    actual = 1.0 if is_success else 0.0
    new_surprise = calculate_surprise(actual, base_rate, scale_factor)
    return update_surprise_ema(old_surprise, new_surprise, alpha)


# Type alias for custom base rate calculators
BaseRateCalculator = Callable[[str], float]


def create_db_base_rate_calculator(
    query_func: Callable[[str], tuple[int, int]],
) -> BaseRateCalculator:
    """
    Create a base rate calculator that queries a database.

    Args:
        query_func: Function that takes category and returns (success_count, failure_count)

    Returns:
        Callable that calculates base rate for a category

    Example:
        def query_db(category: str) -> tuple[int, int]:
            cursor.execute("SELECT success_count, failure_count FROM stats WHERE cat = ?", (category,))
            row = cursor.fetchone()
            return (row[0], row[1]) if row else (0, 0)

        calc_base_rate = create_db_base_rate_calculator(query_db)
        rate = calc_base_rate("type_errors")  # Queries DB and calculates rate
    """

    def calculator(category: str) -> float:
        success, failure = query_func(category)
        return calculate_base_rate(success, failure)

    return calculator
