"""
ASCoT: Adaptive Self-Correction Chain-of-Thought Fragility Analysis.

Based on: https://arxiv.org/pdf/2508.05282

Key insight: Late-stage errors in reasoning chains are significantly
more impactful than early-stage errors. Aragora debates follow similar
patterns - later rounds build on earlier conclusions, amplifying errors.

This module provides fragility analysis to:
1. Identify rounds with high error-compounding risk
2. Recommend increased verification scrutiny for late-stage rounds
3. Integrate with stability detection to gate premature stopping
"""

from dataclasses import dataclass
from typing import Optional, Any
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FragilityScore:
    """Fragility assessment for a debate round."""

    round_number: int
    total_rounds: int
    base_fragility: float  # Position-based fragility (higher for later rounds)
    dependency_depth: int  # How many prior rounds this depends on
    error_risk: float  # Compound error probability
    scrutiny_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    combined_fragility: float = 0.0  # Overall fragility score


@dataclass
class FragilityConfig:
    """Configuration for fragility analysis."""

    lambda_factor: float = 2.0  # Steepness of fragility curve
    base_error_rate: float = 0.05  # Assumed per-step error rate
    critical_threshold: float = 0.8  # Fragility threshold for CRITICAL
    high_threshold: float = 0.6  # Fragility threshold for HIGH
    medium_threshold: float = 0.3  # Fragility threshold for MEDIUM
    dependency_weight: float = 0.4  # Weight for dependency-based risk
    position_weight: float = 0.6  # Weight for position-based fragility


class ASCoTFragilityAnalyzer:
    """
    Analyzes debate rounds for late-stage fragility.

    Applies exponential weighting to later rounds:
    fragility(r) = 1 - exp(-lambda * r / total_rounds)

    Where lambda controls the steepness of the fragility curve.

    Example:
        analyzer = ASCoTFragilityAnalyzer()

        # Calculate fragility for round 5 of 10
        fragility = analyzer.calculate_round_fragility(
            round_number=5,
            total_rounds=10,
            dependencies=[1, 2, 4],  # Depends on rounds 1, 2, 4
        )

        # Get verification config based on fragility
        config = analyzer.get_verification_intensity(fragility)
        if config["formal_verification"]:
            await run_formal_verification(round_content)
    """

    # Scrutiny level configurations for verification phase
    SCRUTINY_CONFIGS: dict[str, dict[str, Any]] = {
        "LOW": {
            "formal_verification": False,
            "evidence_check": False,
            "critique_weight_boost": 1.0,
            "timeout_seconds": 30,
            "require_multi_agent_agreement": False,
        },
        "MEDIUM": {
            "formal_verification": False,
            "evidence_check": True,
            "critique_weight_boost": 1.2,
            "timeout_seconds": 60,
            "require_multi_agent_agreement": False,
        },
        "HIGH": {
            "formal_verification": True,
            "evidence_check": True,
            "critique_weight_boost": 1.5,
            "timeout_seconds": 120,
            "require_multi_agent_agreement": False,
        },
        "CRITICAL": {
            "formal_verification": True,
            "evidence_check": True,
            "critique_weight_boost": 2.0,
            "timeout_seconds": 180,
            "require_multi_agent_agreement": True,
        },
    }

    def __init__(self, config: Optional[FragilityConfig] = None):
        """Initialize the fragility analyzer.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or FragilityConfig()
        self._fragility_history: list[FragilityScore] = []

    def calculate_round_fragility(
        self,
        round_number: int,
        total_rounds: int,
        dependencies: Optional[list[int]] = None,
    ) -> FragilityScore:
        """
        Calculate fragility score for a specific debate round.

        Args:
            round_number: Current round (1-indexed)
            total_rounds: Expected total rounds
            dependencies: List of prior round numbers this round depends on

        Returns:
            FragilityScore with recommended scrutiny level
        """
        if total_rounds <= 0:
            total_rounds = 1  # Prevent division by zero

        if round_number <= 0:
            round_number = 1

        # Base fragility: exponential increase toward end
        # Using: 1 - exp(-lambda * position)
        normalized_position = round_number / total_rounds
        base_fragility = 1.0 - np.exp(-self.config.lambda_factor * normalized_position)

        # Dependency depth increases risk
        # If no explicit dependencies, assume depends on all prior rounds
        if dependencies is None:
            dependency_depth = max(0, round_number - 1)
        else:
            dependency_depth = len(dependencies)

        # Compound error probability using chain rule
        # P(error) = 1 - (1 - base_error_rate)^dependency_depth
        error_risk = 1.0 - (1.0 - self.config.base_error_rate) ** dependency_depth

        # Combined fragility: weighted sum of position-based and dependency-based
        combined_fragility = (
            self.config.position_weight * base_fragility
            + self.config.dependency_weight * error_risk
        )

        # Determine scrutiny level based on combined fragility
        scrutiny_level = self._determine_scrutiny_level(combined_fragility)

        fragility = FragilityScore(
            round_number=round_number,
            total_rounds=total_rounds,
            base_fragility=float(base_fragility),
            dependency_depth=dependency_depth,
            error_risk=float(error_risk),
            scrutiny_level=scrutiny_level,
            combined_fragility=float(combined_fragility),
        )

        # Store in history for telemetry
        self._fragility_history.append(fragility)

        logger.debug(
            "fragility_calculated round=%d/%d base=%.3f deps=%d "
            "error_risk=%.3f combined=%.3f scrutiny=%s",
            round_number,
            total_rounds,
            base_fragility,
            dependency_depth,
            error_risk,
            combined_fragility,
            scrutiny_level,
        )

        return fragility

    def _determine_scrutiny_level(self, combined_fragility: float) -> str:
        """Determine scrutiny level from combined fragility score."""
        if combined_fragility >= self.config.critical_threshold:
            return "CRITICAL"
        elif combined_fragility >= self.config.high_threshold:
            return "HIGH"
        elif combined_fragility >= self.config.medium_threshold:
            return "MEDIUM"
        else:
            return "LOW"

    def get_verification_intensity(
        self,
        fragility: FragilityScore,
    ) -> dict[str, Any]:
        """
        Get verification parameters based on fragility.

        Returns config suitable for consensus verification phase:
        - formal_verification: Whether to run Z3/Lean4 checks
        - evidence_check: Whether to verify evidence grounding
        - critique_weight_boost: Multiplier for critique weights
        - timeout_seconds: Max time for verification
        - require_multi_agent_agreement: Whether multiple agents must agree

        Args:
            fragility: FragilityScore from calculate_round_fragility

        Returns:
            Dict with verification configuration
        """
        base_config = self.SCRUTINY_CONFIGS.get(
            fragility.scrutiny_level,
            self.SCRUTINY_CONFIGS["MEDIUM"],
        )

        # Return a copy to avoid mutation
        config = base_config.copy()

        # Add fragility context
        config["fragility_score"] = fragility.combined_fragility
        config["round_number"] = fragility.round_number

        return config

    def is_in_fragile_zone(
        self,
        round_number: int,
        total_rounds: int,
        threshold: float = 0.6,
    ) -> bool:
        """
        Check if a round is in the "fragile zone" (late-stage high risk).

        Useful for integration with stability detection - if in fragile zone,
        stability detector should be more conservative about stopping.

        Args:
            round_number: Current round number
            total_rounds: Total expected rounds
            threshold: Fragility threshold for "fragile zone"

        Returns:
            True if round is in fragile zone
        """
        fragility = self.calculate_round_fragility(
            round_number=round_number,
            total_rounds=total_rounds,
        )
        return fragility.combined_fragility >= threshold

    def get_fragility_for_stability_gate(
        self,
        round_number: int,
        total_rounds: int,
    ) -> float:
        """
        Get fragility score for use as stability gate.

        This method returns just the combined fragility score, suitable
        for passing to BetaBinomialStabilityDetector.update() as ascot_fragility.

        Args:
            round_number: Current round number
            total_rounds: Total expected rounds

        Returns:
            Combined fragility score (0.0-1.0)
        """
        fragility = self.calculate_round_fragility(
            round_number=round_number,
            total_rounds=total_rounds,
        )
        return fragility.combined_fragility

    def reset(self) -> None:
        """Reset analyzer state for a new debate."""
        self._fragility_history.clear()

    def get_metrics(self) -> dict[str, Any]:
        """Get analyzer metrics for telemetry."""
        if not self._fragility_history:
            return {
                "total_rounds_analyzed": 0,
                "avg_fragility": 0.0,
                "max_fragility": 0.0,
                "critical_rounds": 0,
                "high_rounds": 0,
            }

        fragilities = [f.combined_fragility for f in self._fragility_history]
        scrutiny_counts: dict[str, int] = {}
        for f in self._fragility_history:
            scrutiny_counts[f.scrutiny_level] = scrutiny_counts.get(f.scrutiny_level, 0) + 1

        return {
            "total_rounds_analyzed": len(self._fragility_history),
            "avg_fragility": float(np.mean(fragilities)),
            "max_fragility": float(np.max(fragilities)),
            "min_fragility": float(np.min(fragilities)),
            "critical_rounds": scrutiny_counts.get("CRITICAL", 0),
            "high_rounds": scrutiny_counts.get("HIGH", 0),
            "medium_rounds": scrutiny_counts.get("MEDIUM", 0),
            "low_rounds": scrutiny_counts.get("LOW", 0),
        }


# Convenience function for quick fragility checks
def calculate_fragility(
    round_number: int,
    total_rounds: int,
    lambda_factor: float = 2.0,
) -> float:
    """
    Quick calculation of fragility score without full analyzer.

    Args:
        round_number: Current round (1-indexed)
        total_rounds: Expected total rounds
        lambda_factor: Steepness of fragility curve

    Returns:
        Fragility score (0.0-1.0)
    """
    if total_rounds <= 0:
        return 0.0

    normalized_position = round_number / total_rounds
    base_fragility = 1.0 - np.exp(-lambda_factor * normalized_position)

    # Add simple dependency-based risk
    dependency_depth = max(0, round_number - 1)
    error_risk = 1.0 - (0.95**dependency_depth)  # 5% per-step error

    return float(0.6 * base_fragility + 0.4 * error_risk)


def create_fragility_analyzer(
    lambda_factor: float = 2.0,
    critical_threshold: float = 0.8,
    **kwargs: Any,
) -> ASCoTFragilityAnalyzer:
    """
    Create a fragility analyzer with common configuration.

    Args:
        lambda_factor: Steepness of fragility curve
        critical_threshold: Threshold for CRITICAL scrutiny
        **kwargs: Additional config options

    Returns:
        Configured ASCoTFragilityAnalyzer
    """
    config = FragilityConfig(
        lambda_factor=lambda_factor,
        critical_threshold=critical_threshold,
        **kwargs,
    )
    return ASCoTFragilityAnalyzer(config)
