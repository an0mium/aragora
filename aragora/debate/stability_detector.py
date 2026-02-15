"""
Adaptive Stability Detection via Beta-Binomial Mixture Model.

Based on: https://arxiv.org/abs/2510.12697 (Multi-Agent Debate for LLM Judges)

This module implements adaptive stopping for debates by detecting when consensus
has statistically stabilized, reducing compute costs without sacrificing quality.

Key insight: Use KS-distance between consecutive vote distributions to detect
when further debate rounds are unlikely to change the outcome.
"""

from dataclasses import dataclass
from typing import Any
from collections.abc import Iterable
import logging

import numpy as np

try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


@dataclass
class StabilityResult:
    """Result of stability detection for a debate round."""

    is_stable: bool
    stability_score: float  # 0.0-1.0, higher = more stable
    ks_distance: float  # KS-distance between vote distributions
    rounds_since_stable: int  # How many consecutive rounds stability has held
    recommendation: str  # "stop", "continue", "one_more_round"
    muse_gated: bool = False  # True if MUSE disagreement blocked stopping
    ascot_gated: bool = False  # True if ASCoT fragility blocked stopping
    beta_binomial_prob: float = 0.0  # Beta-Binomial stability probability


@dataclass
class AgreementStabilityResult:
    """Backward-compatible agreement stability result shape."""

    stability: float
    agreement_score: float
    successes: int
    total: int


@dataclass
class StabilityConfig:
    """Configuration for stability detection."""

    stability_threshold: float = 0.85  # Probability threshold for stability
    ks_threshold: float = 0.1  # Max KS-distance to consider stable
    min_stable_rounds: int = 1  # Minimum rounds stability must hold
    muse_disagreement_gate: float = 0.4  # Max MUSE divergence to allow stop
    ascot_fragility_gate: float = 0.7  # Max fragility to allow stop
    min_rounds_before_check: int = 2  # Don't check stability before this round
    alpha_prior: float = 1.0  # Beta distribution prior alpha
    beta_prior: float = 1.0  # Beta distribution prior beta


class BetaBinomialStabilityDetector:
    """
    Detects consensus stability using Beta-Binomial mixture model.

    The detector tracks vote distributions across rounds and uses statistical
    tests to determine when the debate has reached a stable consensus.

    Key features:
    - KS-distance between consecutive vote distributions
    - Beta-Binomial model for stability probability
    - Integration with MUSE (blocks if disagreement too high)
    - Integration with ASCoT (blocks if in late-stage fragile zone)

    Example:
        detector = BetaBinomialStabilityDetector()

        for round_num in range(1, max_rounds + 1):
            # Run debate round, collect votes
            votes = await run_debate_round(round_num)

            # Check stability
            result = detector.update(
                round_votes=votes,
                round_num=round_num,
                muse_divergence=muse_result.divergence_score,
                ascot_fragility=fragility_score,
            )

            if result.recommendation == "stop":
                break
    """

    def __init__(
        self,
        config: StabilityConfig | None = None,
        agreement_threshold: float | None = None,
        alpha_prior: float | None = None,
        beta_prior: float | None = None,
    ):
        """Initialize the stability detector.

        Args:
            config: Configuration options. Uses defaults if not provided.
            agreement_threshold: Optional threshold for agreement-score success.
            alpha_prior: Optional alpha prior for compatibility calculations.
            beta_prior: Optional beta prior for compatibility calculations.
        """
        self.config = config or StabilityConfig()
        self.agreement_threshold = agreement_threshold if agreement_threshold is not None else 0.75
        self.alpha_prior = alpha_prior if alpha_prior is not None else self.config.alpha_prior
        self.beta_prior = beta_prior if beta_prior is not None else self.config.beta_prior
        self._vote_history: list[dict[str, float]] = []
        self._stable_since: int | None = None
        self._stability_scores: list[float] = []

    def calculate_stability(self, agreement_scores: Iterable[float]) -> AgreementStabilityResult:
        """
        Compute a simple beta-binomial posterior over agreement scores.

        This method is kept for compatibility with existing callers in
        ConsensusEstimator and unit tests.
        """
        scores = list(agreement_scores)
        if not scores:
            return AgreementStabilityResult(
                stability=0.0,
                agreement_score=0.0,
                successes=0,
                total=0,
            )

        successes = sum(1 for score in scores if score >= self.agreement_threshold)
        total = len(scores)
        posterior_mean = (self.alpha_prior + successes) / (
            self.alpha_prior + self.beta_prior + total
        )
        return AgreementStabilityResult(
            stability=posterior_mean,
            agreement_score=scores[-1],
            successes=successes,
            total=total,
        )

    def update(
        self,
        round_votes: dict[str, float],
        round_num: int,
        muse_divergence: float | None = None,
        ascot_fragility: float | None = None,
    ) -> StabilityResult:
        """
        Update with new round votes and check stability.

        Args:
            round_votes: Mapping of agent_id to vote weight for winner
            round_num: Current round number (1-indexed)
            muse_divergence: Optional MUSE JSD score (0-1, lower = more agreement)
            ascot_fragility: Optional ASCoT fragility score (0-1, higher = more fragile)

        Returns:
            StabilityResult with recommendation
        """
        self._vote_history.append(round_votes.copy())

        # Not enough history for stability check
        if len(self._vote_history) < 2:
            return StabilityResult(
                is_stable=False,
                stability_score=0.0,
                ks_distance=1.0,
                rounds_since_stable=0,
                recommendation="continue",
                muse_gated=False,
                ascot_gated=False,
                beta_binomial_prob=0.0,
            )

        # Don't check before minimum rounds
        if round_num < self.config.min_rounds_before_check:
            return StabilityResult(
                is_stable=False,
                stability_score=0.0,
                ks_distance=1.0,
                rounds_since_stable=0,
                recommendation="continue",
                muse_gated=False,
                ascot_gated=False,
                beta_binomial_prob=0.0,
            )

        # Calculate KS-distance between last two rounds
        ks_distance = self._calculate_ks_distance(
            self._vote_history[-2],
            self._vote_history[-1],
        )

        # Calculate Beta-Binomial stability probability
        beta_prob = self._calculate_beta_binomial_probability()

        # Calculate overall stability score
        stability_score = self._calculate_stability_score(ks_distance, beta_prob)
        self._stability_scores.append(stability_score)

        # Check if stable
        is_stable = (
            ks_distance < self.config.ks_threshold
            and stability_score >= self.config.stability_threshold
        )

        # Track consecutive stable rounds
        if is_stable:
            if self._stable_since is None:
                self._stable_since = round_num
            rounds_since_stable = round_num - self._stable_since + 1
        else:
            self._stable_since = None
            rounds_since_stable = 0

        # Check gates
        muse_gated = (
            muse_divergence is not None and muse_divergence > self.config.muse_disagreement_gate
        )
        ascot_gated = (
            ascot_fragility is not None and ascot_fragility > self.config.ascot_fragility_gate
        )

        # Determine recommendation
        recommendation = self._determine_recommendation(
            is_stable=is_stable,
            rounds_since_stable=rounds_since_stable,
            muse_gated=muse_gated,
            ascot_gated=ascot_gated,
        )

        logger.debug(
            "stability_check round=%d ks=%.3f score=%.3f stable=%s rec=%s "
            "muse_gate=%s ascot_gate=%s",
            round_num,
            ks_distance,
            stability_score,
            is_stable,
            recommendation,
            muse_gated,
            ascot_gated,
        )

        return StabilityResult(
            is_stable=is_stable,
            stability_score=stability_score,
            ks_distance=ks_distance,
            rounds_since_stable=rounds_since_stable,
            recommendation=recommendation,
            muse_gated=muse_gated,
            ascot_gated=ascot_gated,
            beta_binomial_prob=beta_prob,
        )

    def _calculate_ks_distance(
        self,
        votes1: dict[str, float],
        votes2: dict[str, float],
    ) -> float:
        """Calculate KS-distance between two vote distributions."""
        # Get union of all agents
        all_agents = set(votes1.keys()) | set(votes2.keys())

        if not all_agents:
            return 0.0

        # Convert to arrays with same ordering
        dist1 = np.array([votes1.get(a, 0.0) for a in sorted(all_agents)])
        dist2 = np.array([votes2.get(a, 0.0) for a in sorted(all_agents)])

        # Normalize to probability distributions
        dist1 = self._normalize_distribution(dist1)
        dist2 = self._normalize_distribution(dist2)

        if HAS_SCIPY:
            # Use scipy's KS test
            ks_stat, _ = stats.ks_2samp(dist1, dist2)
            return float(ks_stat)
        else:
            # Fallback: simple max absolute difference of CDFs
            cdf1 = np.cumsum(dist1)
            cdf2 = np.cumsum(dist2)
            return float(np.max(np.abs(cdf1 - cdf2)))

    def _normalize_distribution(self, dist: np.ndarray) -> np.ndarray:
        """Normalize array to valid probability distribution."""
        dist = np.clip(dist, 0, None)  # Ensure non-negative
        total = dist.sum()
        if total == 0:
            # Uniform distribution if all zeros
            return np.ones_like(dist) / len(dist)
        return dist / total

    def _calculate_beta_binomial_probability(self) -> float:
        """Calculate stability probability using Beta-Binomial model.

        Uses the history of stability scores to estimate the probability
        that the debate has converged.
        """
        if len(self._stability_scores) < 2:
            return 0.5  # Uninformative prior

        # Count "stable" outcomes (score > 0.7 as proxy)
        stable_threshold = 0.7
        successes = sum(1 for s in self._stability_scores if s > stable_threshold)
        failures = len(self._stability_scores) - successes

        # Beta posterior
        alpha_post = self.config.alpha_prior + successes
        beta_post = self.config.beta_prior + failures

        if HAS_SCIPY:
            # Probability that true rate > threshold
            return float(1.0 - stats.beta.cdf(stable_threshold, alpha_post, beta_post))
        else:
            # Approximate with point estimate
            mean = alpha_post / (alpha_post + beta_post)
            return float(mean)

    def _calculate_stability_score(
        self,
        ks_distance: float,
        beta_prob: float,
    ) -> float:
        """Calculate combined stability score.

        Combines KS-distance (lower = more stable) with Beta-Binomial probability.
        """
        # KS contribution: inverse distance, capped
        ks_score = 1.0 - min(ks_distance, 1.0)

        # Weighted combination
        combined = 0.6 * ks_score + 0.4 * beta_prob

        return float(combined)

    def _determine_recommendation(
        self,
        is_stable: bool,
        rounds_since_stable: int,
        muse_gated: bool,
        ascot_gated: bool,
    ) -> str:
        """Determine stopping recommendation."""
        if not is_stable:
            return "continue"

        if muse_gated:
            # MUSE shows high disagreement despite stable votes
            logger.debug("stability gated by MUSE divergence")
            return "continue"

        if ascot_gated:
            # ASCoT shows we're in a fragile late-stage zone
            logger.debug("stability gated by ASCoT fragility")
            return "one_more_round"

        if rounds_since_stable >= self.config.min_stable_rounds:
            return "stop"

        return "one_more_round"

    def reset(self) -> None:
        """Reset detector state for a new debate."""
        self._vote_history.clear()
        self._stable_since = None
        self._stability_scores.clear()

    def get_metrics(self) -> dict[str, Any]:
        """Get detector metrics for telemetry."""
        return {
            "total_rounds": len(self._vote_history),
            "stability_scores": self._stability_scores.copy(),
            "avg_stability": (
                float(np.mean(self._stability_scores)) if self._stability_scores else 0.0
            ),
            "stable_since": self._stable_since,
        }


# Convenience function for integration with ConsensusEstimator
def create_stability_detector(
    early_termination_threshold: float = 0.85,
    min_rounds: int = 2,
    **kwargs: Any,
) -> BetaBinomialStabilityDetector:
    """Create a stability detector with common configuration.

    This is a convenience function for integration with the existing
    ConsensusEstimator infrastructure.

    Args:
        early_termination_threshold: Maps to stability_threshold
        min_rounds: Maps to min_rounds_before_check
        **kwargs: Additional config options

    Returns:
        Configured BetaBinomialStabilityDetector
    """
    config = StabilityConfig(
        stability_threshold=early_termination_threshold,
        min_rounds_before_check=min_rounds,
        **kwargs,
    )
    return BetaBinomialStabilityDetector(config)
