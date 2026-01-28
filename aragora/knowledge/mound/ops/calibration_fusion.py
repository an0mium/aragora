"""
Multi-Party Calibration Fusion for Knowledge Mound Phase A3.

This module provides algorithms for aggregating calibration predictions from
multiple agents into consensus scores with proper uncertainty quantification.

Key Components:
- AgentPrediction: Individual agent's prediction with confidence
- CalibrationConsensus: Result of fusing multiple predictions
- CalibrationFusionEngine: Main engine for multi-party calibration fusion

Features:
- Weighted averaging based on agent calibration accuracy
- Krippendorff's alpha for inter-rater agreement
- Outlier detection using modified Z-scores
- Consensus strength calculation

Usage:
    from aragora.knowledge.mound.ops.calibration_fusion import (
        CalibrationFusionEngine,
        AgentPrediction,
        CalibrationFusionStrategy,
    )

    engine = CalibrationFusionEngine()
    predictions = [
        AgentPrediction("claude", 0.8, "winner_a"),
        AgentPrediction("gpt-4", 0.75, "winner_a"),
        AgentPrediction("gemini", 0.6, "winner_b"),
    ]
    consensus = engine.fuse_predictions(
        predictions,
        weights={"claude": 0.9, "gpt-4": 0.85, "gemini": 0.7},
    )
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CalibrationFusionStrategy(Enum):
    """Strategy for fusing calibration predictions."""

    WEIGHTED_AVERAGE = "weighted_average"
    """Weighted average by calibration accuracy."""

    RELIABILITY_WEIGHTED = "reliability_weighted"
    """Weight by historical reliability scores."""

    BAYESIAN = "bayesian"
    """Bayesian updating of prior beliefs."""

    MEDIAN = "median"
    """Use median prediction (robust to outliers)."""

    TRIMMED_MEAN = "trimmed_mean"
    """Trimmed mean excluding outliers."""

    CONSENSUS_ONLY = "consensus_only"
    """Only use predictions that agree with majority."""


@dataclass
class AgentPrediction:
    """A calibration prediction from a single agent."""

    agent_name: str
    """Name of the predicting agent."""

    confidence: float
    """Confidence in the prediction (0.0 to 1.0)."""

    predicted_outcome: str
    """The predicted outcome (e.g., winner agent name)."""

    domain: Optional[str] = None
    """Optional domain context for the prediction."""

    calibration_accuracy: float = 0.5
    """Historical calibration accuracy of this agent."""

    brier_score: float = 0.25
    """Historical Brier score (lower is better, 0-1 range)."""

    prediction_count: int = 0
    """Number of predictions this agent has made."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When the prediction was made."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional prediction metadata."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "agent_name": self.agent_name,
            "confidence": self.confidence,
            "predicted_outcome": self.predicted_outcome,
            "domain": self.domain,
            "calibration_accuracy": self.calibration_accuracy,
            "brier_score": self.brier_score,
            "prediction_count": self.prediction_count,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class CalibrationConsensus:
    """Result of fusing multiple calibration predictions."""

    debate_id: str
    """ID of the debate or event being predicted."""

    predictions: List[AgentPrediction]
    """All predictions that were fused."""

    fused_confidence: float
    """Weighted average confidence (0.0 to 1.0)."""

    predicted_outcome: str
    """The consensus predicted outcome."""

    consensus_strength: float
    """Strength of consensus (0.0 to 1.0, higher is stronger)."""

    agreement_ratio: float
    """Ratio of agents agreeing with majority (0.0 to 1.0)."""

    disagreement_score: float
    """Variance-based disagreement measure (0.0+, lower is better)."""

    krippendorff_alpha: float
    """Inter-rater agreement statistic (-1.0 to 1.0)."""

    strategy_used: CalibrationFusionStrategy = CalibrationFusionStrategy.WEIGHTED_AVERAGE
    """Fusion strategy that was applied."""

    outliers_detected: List[str] = field(default_factory=list)
    """Agent names flagged as outliers."""

    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    """95% confidence interval for fused confidence."""

    participating_agents: List[str] = field(default_factory=list)
    """Agents that contributed predictions."""

    weights_used: Dict[str, float] = field(default_factory=dict)
    """Weights applied to each agent."""

    fused_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When the fusion was performed."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional fusion metadata."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "debate_id": self.debate_id,
            "predictions": [p.to_dict() for p in self.predictions],
            "fused_confidence": self.fused_confidence,
            "predicted_outcome": self.predicted_outcome,
            "consensus_strength": self.consensus_strength,
            "agreement_ratio": self.agreement_ratio,
            "disagreement_score": self.disagreement_score,
            "krippendorff_alpha": self.krippendorff_alpha,
            "strategy_used": self.strategy_used.value,
            "outliers_detected": self.outliers_detected,
            "confidence_interval": self.confidence_interval,
            "participating_agents": self.participating_agents,
            "weights_used": self.weights_used,
            "fused_at": self.fused_at.isoformat(),
            "metadata": self.metadata,
        }

    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high-confidence consensus."""
        return (
            self.consensus_strength >= 0.7
            and self.agreement_ratio >= 0.7
            and len(self.outliers_detected) == 0
        )

    @property
    def needs_review(self) -> bool:
        """Check if this consensus should be reviewed."""
        return (
            self.consensus_strength < 0.5
            or self.disagreement_score > 0.2
            or self.krippendorff_alpha < 0.4
        )


@dataclass
class FusionConfig:
    """Configuration for calibration fusion."""

    min_predictions: int = 2
    """Minimum predictions required for fusion."""

    outlier_threshold: float = 2.0
    """Modified Z-score threshold for outlier detection."""

    trim_percent: float = 0.1
    """Percent to trim from each end for TRIMMED_MEAN."""

    min_calibration_accuracy: float = 0.0
    """Minimum calibration accuracy to include agent."""

    min_prediction_count: int = 0
    """Minimum predictions an agent must have made."""

    use_brier_weights: bool = True
    """Weight by inverse Brier score (penalize poor calibration)."""

    confidence_level: float = 0.95
    """Confidence level for interval estimation."""


class CalibrationFusionEngine:
    """Engine for fusing multi-party calibration predictions.

    This engine aggregates predictions from multiple agents, detecting
    outliers and computing consensus metrics for reliability assessment.
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        """Initialize the calibration fusion engine.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or FusionConfig()
        self._fusion_history: List[CalibrationConsensus] = []

    def fuse_predictions(
        self,
        predictions: List[AgentPrediction],
        debate_id: str = "",
        weights: Optional[Dict[str, float]] = None,
        strategy: CalibrationFusionStrategy = CalibrationFusionStrategy.WEIGHTED_AVERAGE,
    ) -> CalibrationConsensus:
        """Fuse multiple agent predictions into a consensus.

        Args:
            predictions: List of agent predictions to fuse.
            debate_id: Optional debate ID for tracking.
            weights: Optional explicit weights per agent.
            strategy: Fusion strategy to use.

        Returns:
            CalibrationConsensus with the fused result.
        """
        if len(predictions) < self.config.min_predictions:
            return self._insufficient_predictions_result(predictions, debate_id)

        # Filter predictions by minimum criteria
        valid_predictions = self._filter_predictions(predictions)
        if len(valid_predictions) < self.config.min_predictions:
            return self._insufficient_predictions_result(predictions, debate_id)

        # Compute weights if not provided
        if weights is None:
            weights = self._compute_weights(valid_predictions)

        # Detect outliers
        outliers = self.detect_outliers(valid_predictions)

        # Filter out outliers for certain strategies
        if strategy == CalibrationFusionStrategy.TRIMMED_MEAN:
            valid_predictions = [p for p in valid_predictions if p.agent_name not in outliers]

        # Determine majority outcome
        predicted_outcome, agreement_ratio = self._compute_majority_outcome(valid_predictions)

        # Apply fusion strategy
        fused_confidence = self._apply_strategy(
            valid_predictions, weights, strategy, predicted_outcome
        )

        # Compute consensus metrics
        disagreement_score = self._compute_disagreement(valid_predictions)
        consensus_strength = self._compute_consensus_strength(
            valid_predictions, agreement_ratio, disagreement_score
        )
        krippendorff_alpha = self.compute_krippendorff_alpha(valid_predictions)
        confidence_interval = self._compute_confidence_interval(valid_predictions, weights)

        result = CalibrationConsensus(
            debate_id=debate_id,
            predictions=predictions,
            fused_confidence=fused_confidence,
            predicted_outcome=predicted_outcome,
            consensus_strength=consensus_strength,
            agreement_ratio=agreement_ratio,
            disagreement_score=disagreement_score,
            krippendorff_alpha=krippendorff_alpha,
            strategy_used=strategy,
            outliers_detected=outliers,
            confidence_interval=confidence_interval,
            participating_agents=[p.agent_name for p in valid_predictions],
            weights_used=weights,
        )

        self._fusion_history.append(result)
        return result

    def _filter_predictions(self, predictions: List[AgentPrediction]) -> List[AgentPrediction]:
        """Filter predictions by minimum criteria.

        Args:
            predictions: All predictions.

        Returns:
            Predictions meeting minimum criteria.
        """
        return [
            p
            for p in predictions
            if p.calibration_accuracy >= self.config.min_calibration_accuracy
            and p.prediction_count >= self.config.min_prediction_count
        ]

    def _compute_weights(self, predictions: List[AgentPrediction]) -> Dict[str, float]:
        """Compute weights for each agent based on calibration metrics.

        Args:
            predictions: Predictions to weight.

        Returns:
            Dict mapping agent name to weight.
        """
        weights: Dict[str, float] = {}

        for p in predictions:
            # Base weight from calibration accuracy
            weight = p.calibration_accuracy

            # Adjust by inverse Brier score (1 - brier gives higher weight to lower scores)
            if self.config.use_brier_weights and p.brier_score < 1.0:
                brier_factor = 1.0 - p.brier_score
                weight *= (1.0 + brier_factor) / 2.0

            # Boost for prediction count (experience)
            if p.prediction_count > 0:
                experience_factor = min(1.0, math.log10(p.prediction_count + 1) / 2.0)
                weight *= (1.0 + experience_factor) / 2.0

            weights[p.agent_name] = max(0.1, weight)  # Minimum weight of 0.1

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _compute_majority_outcome(self, predictions: List[AgentPrediction]) -> Tuple[str, float]:
        """Determine the majority predicted outcome.

        Args:
            predictions: All predictions.

        Returns:
            Tuple of (majority_outcome, agreement_ratio).
        """
        outcome_counts: Dict[str, int] = {}
        for p in predictions:
            outcome = p.predicted_outcome
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        if not outcome_counts:
            return "", 0.0

        majority_outcome = max(outcome_counts.keys(), key=lambda k: outcome_counts[k])
        majority_count = outcome_counts[majority_outcome]
        agreement_ratio = majority_count / len(predictions)

        return majority_outcome, agreement_ratio

    def _apply_strategy(
        self,
        predictions: List[AgentPrediction],
        weights: Dict[str, float],
        strategy: CalibrationFusionStrategy,
        majority_outcome: str,
    ) -> float:
        """Apply the fusion strategy to compute fused confidence.

        Args:
            predictions: Predictions to fuse.
            weights: Weight per agent.
            strategy: Fusion strategy.
            majority_outcome: The majority predicted outcome.

        Returns:
            Fused confidence score.
        """
        # Get confidences for majority outcome
        confidences = [p.confidence for p in predictions if p.predicted_outcome == majority_outcome]

        # If no one predicted the majority, use all confidences
        if not confidences:
            confidences = [p.confidence for p in predictions]

        if strategy == CalibrationFusionStrategy.WEIGHTED_AVERAGE:
            # Weighted average
            total_weight = 0.0
            weighted_sum = 0.0
            for p in predictions:
                if p.predicted_outcome == majority_outcome:
                    w = weights.get(p.agent_name, 1.0)
                    weighted_sum += p.confidence * w
                    total_weight += w
            return weighted_sum / total_weight if total_weight > 0 else 0.5

        elif strategy == CalibrationFusionStrategy.RELIABILITY_WEIGHTED:
            # Weight by calibration accuracy directly
            total_weight = 0.0
            weighted_sum = 0.0
            for p in predictions:
                if p.predicted_outcome == majority_outcome:
                    w = p.calibration_accuracy
                    weighted_sum += p.confidence * w
                    total_weight += w
            return weighted_sum / total_weight if total_weight > 0 else 0.5

        elif strategy == CalibrationFusionStrategy.BAYESIAN:
            # Simplified Bayesian: update prior with each prediction
            # Prior: 0.5 (uninformative)
            posterior = 0.5
            for p in predictions:
                if p.predicted_outcome == majority_outcome:
                    # Weight update by calibration accuracy
                    likelihood = p.confidence * p.calibration_accuracy
                    posterior = (posterior * likelihood) / (
                        posterior * likelihood + (1 - posterior) * (1 - likelihood)
                    )
            return posterior

        elif strategy == CalibrationFusionStrategy.MEDIAN:
            return statistics.median(confidences) if confidences else 0.5

        elif strategy == CalibrationFusionStrategy.TRIMMED_MEAN:
            n = len(confidences)
            trim_count = max(1, int(n * self.config.trim_percent))
            if n <= 2 * trim_count:
                return statistics.mean(confidences) if confidences else 0.5
            sorted_conf = sorted(confidences)
            trimmed = sorted_conf[trim_count : n - trim_count]
            return statistics.mean(trimmed) if trimmed else 0.5

        elif strategy == CalibrationFusionStrategy.CONSENSUS_ONLY:
            # Only use predictions matching majority
            matching = [
                p.confidence for p in predictions if p.predicted_outcome == majority_outcome
            ]
            return statistics.mean(matching) if matching else 0.5

        # Default fallback
        return statistics.mean(confidences) if confidences else 0.5

    def _compute_disagreement(self, predictions: List[AgentPrediction]) -> float:
        """Compute disagreement score based on variance.

        Args:
            predictions: All predictions.

        Returns:
            Disagreement score (variance of confidences).
        """
        if len(predictions) < 2:
            return 0.0

        confidences = [p.confidence for p in predictions]
        return statistics.variance(confidences)

    def _compute_consensus_strength(
        self,
        predictions: List[AgentPrediction],
        agreement_ratio: float,
        disagreement_score: float,
    ) -> float:
        """Compute overall consensus strength.

        Combines agreement ratio and inverse disagreement into a single score.

        Args:
            predictions: All predictions.
            agreement_ratio: Ratio of agents agreeing.
            disagreement_score: Variance-based disagreement.

        Returns:
            Consensus strength (0.0 to 1.0).
        """
        # Base strength from agreement
        strength = agreement_ratio

        # Penalize for high disagreement (variance)
        # Max variance for [0,1] is 0.25 (when half are 0 and half are 1)
        variance_penalty = min(1.0, disagreement_score / 0.25)
        strength *= 1.0 - (0.5 * variance_penalty)

        # Boost for number of predictions
        if len(predictions) >= 5:
            strength *= 1.1
        elif len(predictions) >= 3:
            strength *= 1.05

        return min(1.0, max(0.0, strength))

    def compute_krippendorff_alpha(self, predictions: List[AgentPrediction]) -> float:
        """Compute Krippendorff's alpha for inter-rater agreement.

        This is a simplified implementation for ordinal data (confidences).

        Args:
            predictions: All predictions.

        Returns:
            Krippendorff's alpha (-1.0 to 1.0, higher is better).
        """
        if len(predictions) < 2:
            return 1.0  # Perfect agreement with one rater

        # Discretize confidences into buckets for ordinal comparison
        buckets = [int(p.confidence * 10) for p in predictions]

        # Compute observed disagreement
        n = len(buckets)
        observed_disagreement = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                observed_disagreement += (buckets[i] - buckets[j]) ** 2

        observed_disagreement /= n * (n - 1) / 2

        # Compute expected disagreement
        mean_bucket = sum(buckets) / n
        expected_disagreement = sum((b - mean_bucket) ** 2 for b in buckets) / n

        if expected_disagreement == 0:
            return 1.0  # Perfect agreement

        alpha = 1.0 - (observed_disagreement / expected_disagreement)
        return max(-1.0, min(1.0, alpha))

    def detect_outliers(
        self,
        predictions: List[AgentPrediction],
        threshold: Optional[float] = None,
    ) -> List[str]:
        """Detect outlier predictions using modified Z-scores.

        Args:
            predictions: All predictions.
            threshold: Optional Z-score threshold override.

        Returns:
            List of agent names flagged as outliers.
        """
        if len(predictions) < 3:
            return []

        threshold = threshold or self.config.outlier_threshold
        confidences = [p.confidence for p in predictions]

        # Use median absolute deviation (MAD) for robust outlier detection
        median = statistics.median(confidences)
        deviations = [abs(c - median) for c in confidences]
        mad = statistics.median(deviations)

        if mad == 0:
            return []

        # Modified Z-scores
        outliers = []
        for p, c in zip(predictions, confidences):
            modified_z = 0.6745 * (c - median) / mad
            if abs(modified_z) > threshold:
                outliers.append(p.agent_name)

        return outliers

    def _compute_confidence_interval(
        self,
        predictions: List[AgentPrediction],
        weights: Dict[str, float],
    ) -> Tuple[float, float]:
        """Compute confidence interval for fused confidence.

        Uses weighted standard error for interval estimation.

        Args:
            predictions: All predictions.
            weights: Weights per agent.

        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        if len(predictions) < 2:
            return (0.0, 1.0)

        confidences = [p.confidence for p in predictions]
        mean = sum(
            c * weights.get(p.agent_name, 1.0) for c, p in zip(confidences, predictions)
        ) / sum(weights.get(p.agent_name, 1.0) for p in predictions)

        # Weighted standard deviation
        weighted_var = sum(
            weights.get(p.agent_name, 1.0) * (c - mean) ** 2
            for c, p in zip(confidences, predictions)
        ) / sum(weights.get(p.agent_name, 1.0) for p in predictions)

        std_error = math.sqrt(weighted_var / len(predictions))

        # 95% confidence interval (approximate z = 1.96)
        z = 1.96 if self.config.confidence_level == 0.95 else 2.576
        lower = max(0.0, mean - z * std_error)
        upper = min(1.0, mean + z * std_error)

        return (lower, upper)

    def _insufficient_predictions_result(
        self,
        predictions: List[AgentPrediction],
        debate_id: str,
    ) -> CalibrationConsensus:
        """Create result when insufficient predictions available.

        Args:
            predictions: Available predictions.
            debate_id: Debate ID.

        Returns:
            CalibrationConsensus with insufficient data indication.
        """
        if predictions:
            # Use single prediction if available
            p = predictions[0]
            return CalibrationConsensus(
                debate_id=debate_id,
                predictions=predictions,
                fused_confidence=p.confidence,
                predicted_outcome=p.predicted_outcome,
                consensus_strength=0.0,
                agreement_ratio=1.0,
                disagreement_score=0.0,
                krippendorff_alpha=1.0,
                participating_agents=[p.agent_name],
                weights_used={p.agent_name: 1.0},
                metadata={"insufficient_data": True, "reason": "Single prediction only"},
            )

        return CalibrationConsensus(
            debate_id=debate_id,
            predictions=[],
            fused_confidence=0.5,
            predicted_outcome="",
            consensus_strength=0.0,
            agreement_ratio=0.0,
            disagreement_score=0.0,
            krippendorff_alpha=0.0,
            metadata={"insufficient_data": True, "reason": "No predictions"},
        )

    def get_fusion_history(
        self,
        limit: int = 100,
        debate_id: Optional[str] = None,
    ) -> List[CalibrationConsensus]:
        """Get recent fusion history.

        Args:
            limit: Maximum results to return.
            debate_id: Optional filter by debate ID.

        Returns:
            List of recent CalibrationConsensus results.
        """
        history = self._fusion_history

        if debate_id:
            history = [h for h in history if h.debate_id == debate_id]

        return history[-limit:]

    def get_agent_performance(self, agent_name: str) -> Dict[str, Any]:
        """Get performance metrics for an agent across fusions.

        Args:
            agent_name: Name of the agent.

        Returns:
            Dict with performance metrics.
        """
        # Find all fusions where agent participated
        agent_fusions = [f for f in self._fusion_history if agent_name in f.participating_agents]

        if not agent_fusions:
            return {
                "agent_name": agent_name,
                "fusion_count": 0,
                "avg_weight": 0.0,
                "outlier_count": 0,
                "agreement_rate": 0.0,
            }

        total_weight = sum(f.weights_used.get(agent_name, 0.0) for f in agent_fusions)
        outlier_count = sum(1 for f in agent_fusions if agent_name in f.outliers_detected)

        # Check how often agent agreed with majority
        agreements = 0
        for f in agent_fusions:
            for p in f.predictions:
                if p.agent_name == agent_name:
                    if p.predicted_outcome == f.predicted_outcome:
                        agreements += 1
                    break

        return {
            "agent_name": agent_name,
            "fusion_count": len(agent_fusions),
            "avg_weight": total_weight / len(agent_fusions),
            "outlier_count": outlier_count,
            "outlier_rate": outlier_count / len(agent_fusions),
            "agreement_rate": agreements / len(agent_fusions),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get overall fusion statistics.

        Returns:
            Dict with fusion metrics.
        """
        if not self._fusion_history:
            return {
                "total_fusions": 0,
                "avg_consensus_strength": 0.0,
                "avg_agreement_ratio": 0.0,
                "avg_krippendorff_alpha": 0.0,
                "high_confidence_rate": 0.0,
            }

        total = len(self._fusion_history)
        avg_strength = sum(f.consensus_strength for f in self._fusion_history) / total
        avg_agreement = sum(f.agreement_ratio for f in self._fusion_history) / total
        avg_alpha = sum(f.krippendorff_alpha for f in self._fusion_history) / total
        high_conf = sum(1 for f in self._fusion_history if f.is_high_confidence)

        return {
            "total_fusions": total,
            "avg_consensus_strength": avg_strength,
            "avg_agreement_ratio": avg_agreement,
            "avg_krippendorff_alpha": avg_alpha,
            "high_confidence_rate": high_conf / total,
            "needs_review_count": sum(1 for f in self._fusion_history if f.needs_review),
            "by_strategy": self._stats_by_strategy(),
        }

    def _stats_by_strategy(self) -> Dict[str, int]:
        """Get counts by fusion strategy."""
        counts: Dict[str, int] = {}
        for f in self._fusion_history:
            strategy = f.strategy_used.value
            counts[strategy] = counts.get(strategy, 0) + 1
        return counts


# Singleton instance
_calibration_fusion_engine: Optional[CalibrationFusionEngine] = None


def get_calibration_fusion_engine(
    config: Optional[FusionConfig] = None,
) -> CalibrationFusionEngine:
    """Get or create the singleton calibration fusion engine.

    Args:
        config: Optional configuration (only used on first call).

    Returns:
        CalibrationFusionEngine instance.
    """
    global _calibration_fusion_engine
    if _calibration_fusion_engine is None:
        _calibration_fusion_engine = CalibrationFusionEngine(config)
    return _calibration_fusion_engine


__all__ = [
    # Enums
    "CalibrationFusionStrategy",
    # Dataclasses
    "AgentPrediction",
    "CalibrationConsensus",
    "FusionConfig",
    # Engine
    "CalibrationFusionEngine",
    "get_calibration_fusion_engine",
]
