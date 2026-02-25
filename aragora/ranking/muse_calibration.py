"""
MUSE: Multi-LLM Uncertainty via Subset Ensembles.

Based on: https://arxiv.org/abs/2507.07236

This module implements ensemble-level uncertainty quantification using
Jensen-Shannon Divergence to identify well-calibrated model subsets.

Key insight: Well-calibrated model subsets produce more reliable uncertainty
estimates than full ensemble averaging or single-model confidence.
"""

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Sequence
from itertools import combinations
import logging


def _np():
    """Lazy-load numpy to avoid ~3-5s import at module level."""
    import numpy as np

    return np


def _jensenshannon(p: Any, q: Any) -> float:
    """Lazy-loading wrapper for Jensen-Shannon divergence.

    Uses scipy if available, otherwise falls back to a pure-numpy implementation.
    """
    np = _np()
    try:
        from scipy.spatial.distance import jensenshannon as _jsd

        return float(_jsd(p, q))
    except ImportError:
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        eps = 1e-10
        kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
        kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
        return float(np.sqrt(0.5 * kl_pm + 0.5 * kl_qm))


logger = logging.getLogger(__name__)


@dataclass
class MUSEResult:
    """Result from MUSE ensemble uncertainty calculation."""

    consensus_confidence: float  # 0.0-1.0, higher = more agreement
    divergence_score: float  # JSD, lower = more agreement
    best_subset: set[str]  # Agent IDs in best-calibrated subset
    subset_agreement: float  # Agreement within best subset
    subset_brier_score: float  # Average Brier score of subset
    individual_divergences: dict[str, float] = field(default_factory=dict)


@dataclass
class MUSEConfig:
    """Configuration for MUSE calculator."""

    min_subset_size: int = 2  # Minimum agents in subset
    max_subset_size: int = 5  # Maximum agents to consider
    default_brier: float = 0.5  # Default Brier score for unknown agents
    confidence_from_divergence: bool = True  # Derive confidence from JSD


class MUSECalculator:
    """
    Calculates ensemble uncertainty using Jensen-Shannon Divergence.

    The calculator identifies well-calibrated subsets of agents and uses
    their agreement/disagreement to quantify ensemble uncertainty.

    Example:
        calculator = MUSECalculator()

        # After collecting agent responses
        responses = {
            "claude": {"answer": "A", "confidence": 0.9, "distribution": [0.9, 0.1]},
            "gpt-4": {"answer": "A", "confidence": 0.8, "distribution": [0.8, 0.2]},
            "gemini": {"answer": "B", "confidence": 0.7, "distribution": [0.3, 0.7]},
        }

        # Historical calibration (Brier scores, lower = better)
        calibration = {"claude": 0.15, "gpt-4": 0.18, "gemini": 0.25}

        result = calculator.calculate_ensemble_uncertainty(responses, calibration)
        print(f"Consensus confidence: {result.consensus_confidence}")
        print(f"Best subset: {result.best_subset}")
    """

    def __init__(self, config: MUSEConfig | None = None):
        """Initialize MUSE calculator.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or MUSEConfig()
        self._calibration_history: dict[str, list[float]] = {}

    def calculate_ensemble_uncertainty(
        self,
        agent_responses: dict[str, dict[str, Any]],
        historical_calibration: dict[str, float] | None = None,
    ) -> MUSEResult:
        """
        Calculate MUSE uncertainty score for an ensemble of responses.

        Args:
            agent_responses: Mapping of agent_id to response dict containing:
                - answer: The agent's answer
                - confidence: Scalar confidence (0-1)
                - distribution: Optional probability distribution over choices
            historical_calibration: Mapping of agent_id to Brier score (lower = better)

        Returns:
            MUSEResult with consensus confidence and best subset
        """
        if not agent_responses:
            return MUSEResult(
                consensus_confidence=0.0,
                divergence_score=1.0,
                best_subset=set(),
                subset_agreement=0.0,
                subset_brier_score=1.0,
            )

        calibration = historical_calibration or {}

        # Handle small ensembles
        if len(agent_responses) < self.config.min_subset_size:
            avg_confidence = _np().mean(
                [r.get("confidence", 0.5) for r in agent_responses.values()]
            )
            return MUSEResult(
                consensus_confidence=float(avg_confidence),
                divergence_score=0.0,
                best_subset=set(agent_responses.keys()),
                subset_agreement=float(avg_confidence),
                subset_brier_score=self._avg_brier(list(agent_responses.keys()), calibration),
            )

        # Find best-calibrated subset
        best_subset, best_brier = self._find_best_subset(agent_responses, calibration)

        # Calculate JSD for best subset
        divergence_score, individual_divs = self._calculate_subset_jsd(best_subset, agent_responses)

        # Convert JSD to confidence (lower divergence = higher confidence)
        if self.config.confidence_from_divergence:
            consensus_confidence = 1.0 - min(divergence_score, 1.0)
        else:
            # Use average confidence from subset
            consensus_confidence = _np().mean(
                [agent_responses[a].get("confidence", 0.5) for a in best_subset]
            )

        # Calculate subset agreement (inverse of confidence variance)
        subset_confidences = [agent_responses[a].get("confidence", 0.5) for a in best_subset]
        subset_agreement = 1.0 - min(float(_np().std(subset_confidences)), 1.0)

        logger.debug(
            "muse_calculation subset=%s jsd=%.3f confidence=%.3f agreement=%.3f",
            best_subset,
            divergence_score,
            consensus_confidence,
            subset_agreement,
        )

        return MUSEResult(
            consensus_confidence=float(consensus_confidence),
            divergence_score=float(divergence_score),
            best_subset=best_subset,
            subset_agreement=float(subset_agreement),
            subset_brier_score=best_brier,
            individual_divergences=individual_divs,
        )

    def _find_best_subset(
        self,
        responses: dict[str, dict[str, Any]],
        calibration: dict[str, float],
    ) -> tuple[set[str], float]:
        """Find the best-calibrated subset of agents.

        Uses historical Brier scores to identify agents with the best
        calibration track record.

        Args:
            responses: Agent responses
            calibration: Historical Brier scores

        Returns:
            Tuple of (best subset, average Brier score)
        """
        agents = list(responses.keys())
        best_subset: set[str] = set(agents[: self.config.min_subset_size])
        best_score = float("inf")

        # Enumerate subsets of different sizes
        for size in range(
            self.config.min_subset_size,
            min(len(agents), self.config.max_subset_size) + 1,
        ):
            for subset in combinations(agents, size):
                # Calculate average Brier score for subset
                avg_brier = self._avg_brier(list(subset), calibration)

                # Also consider answer agreement as tiebreaker
                agreement_bonus = self._answer_agreement_bonus(subset, responses)
                combined_score = avg_brier - 0.1 * agreement_bonus

                if combined_score < best_score:
                    best_score = combined_score
                    best_subset = set(subset)

        return best_subset, self._avg_brier(list(best_subset), calibration)

    def _avg_brier(
        self,
        agents: Sequence[str],
        calibration: dict[str, float],
    ) -> float:
        """Calculate average Brier score for agents."""
        scores = [calibration.get(a, self.config.default_brier) for a in agents]
        return float(_np().mean(scores))

    def _answer_agreement_bonus(
        self,
        subset: tuple[str, ...],
        responses: dict[str, dict[str, Any]],
    ) -> float:
        """Calculate bonus for agents that agree on answer."""
        answers = [responses[a].get("answer") for a in subset]
        if not answers:
            return 0.0

        # Count most common answer
        from collections import Counter

        counts = Counter(a for a in answers if a is not None)
        if not counts:
            return 0.0

        most_common_count = counts.most_common(1)[0][1]
        return most_common_count / len(answers)

    def _calculate_subset_jsd(
        self,
        subset: set[str],
        responses: dict[str, dict[str, Any]],
    ) -> tuple[float, dict[str, float]]:
        """Calculate average pairwise JSD for subset.

        Args:
            subset: Agent IDs in subset
            responses: All agent responses

        Returns:
            Tuple of (average JSD, individual divergences from mean)
        """
        agents = list(subset)
        if len(agents) < 2:
            return 0.0, {}

        # Get distributions, ensuring consistent length
        distributions = []
        for agent_id in agents:
            response = responses[agent_id]
            dist = response.get("distribution")
            if dist is None:
                # Create binary distribution from confidence
                conf = response.get("confidence", 0.5)
                dist = [conf, 1 - conf]
            distributions.append(_np().array(dist, dtype=float))

        # Normalize to same length
        max_len = max(len(d) for d in distributions)
        normalized = [self._normalize_distribution(d, max_len) for d in distributions]

        # Calculate pairwise JSD
        jsd_scores = []
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                jsd = _jensenshannon(normalized[i], normalized[j])
                jsd_scores.append(jsd)

        avg_jsd = float(_np().mean(jsd_scores)) if jsd_scores else 0.0

        # Calculate individual divergences from mean distribution
        mean_dist = _np().mean(normalized, axis=0)
        individual_divs = {
            agent_id: float(_jensenshannon(normalized[i], mean_dist))
            for i, agent_id in enumerate(agents)
        }

        return avg_jsd, individual_divs

    def _normalize_distribution(
        self,
        dist: Any,
        target_len: int,
    ) -> Any:
        """Normalize distribution to target length with valid probabilities."""
        np = _np()
        dist = np.array(dist, dtype=float)

        if len(dist) == target_len:
            arr = dist
        elif len(dist) == 1:
            # Single confidence value -> binary distribution
            conf = float(dist[0])
            arr = np.zeros(target_len)
            arr[0] = conf
            arr[1] = 1 - conf if target_len > 1 else 0
        elif len(dist) < target_len:
            # Pad with zeros
            arr = np.zeros(target_len)
            arr[: len(dist)] = dist
        else:
            # Truncate (shouldn't happen with proper usage)
            arr = dist[:target_len]

        # Ensure valid probability distribution
        arr = np.clip(arr, 1e-10, 1.0)
        return arr / arr.sum()

    def update_calibration(
        self,
        agent_id: str,
        predicted_confidence: float,
        actual_outcome: float,
    ) -> None:
        """Update calibration history with new observation.

        Args:
            agent_id: The agent that made the prediction
            predicted_confidence: The confidence the agent expressed
            actual_outcome: 1.0 if correct, 0.0 if incorrect
        """
        brier = (predicted_confidence - actual_outcome) ** 2

        if agent_id not in self._calibration_history:
            self._calibration_history[agent_id] = []

        self._calibration_history[agent_id].append(brier)

    def get_calibration_scores(self) -> dict[str, float]:
        """Get current calibration scores from history.

        Returns:
            Mapping of agent_id to average Brier score
        """
        return {
            agent_id: float(_np().mean(scores))
            for agent_id, scores in self._calibration_history.items()
            if scores
        }

    def reset_history(self) -> None:
        """Reset calibration history."""
        self._calibration_history.clear()


def apply_muse_to_votes(
    votes: list[dict[str, Any]],
    muse_result: MUSEResult,
    muse_weight: float = 0.15,
) -> list[dict[str, Any]]:
    """Apply MUSE adjustment to vote weights.

    Boosts weights for agents in the best-calibrated subset.

    Args:
        votes: List of vote dicts with 'agent_id' and 'weight' fields
        muse_result: Result from MUSE calculation
        muse_weight: Weight factor for MUSE boost

    Returns:
        Votes with adjusted weights
    """
    adjusted = []
    for vote in votes:
        vote_copy = vote.copy()
        agent_id = vote.get("agent_id")

        if agent_id in muse_result.best_subset:
            # Boost weight for well-calibrated subset members
            current_weight = vote.get("weight", 1.0)
            boost = 1.0 + muse_weight * muse_result.subset_agreement
            vote_copy["weight"] = current_weight * boost
            vote_copy["muse_boosted"] = True
        else:
            vote_copy["muse_boosted"] = False

        adjusted.append(vote_copy)

    return adjusted
