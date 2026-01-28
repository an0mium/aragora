"""
Quality Signals Module for Knowledge Mound Phase A3.

This module provides calibration-driven quality metrics for knowledge items.
It computes calibrated confidence based on contributor accuracy, detects
overconfidence patterns, and evaluates source reliability.

Key Components:
- QualitySignals: Quality metrics for a knowledge item
- SourceReliability: Historical accuracy for a knowledge source
- QualitySignalEngine: Main quality computation engine

Usage:
    from aragora.knowledge.mound.ops.quality_signals import (
        QualitySignalEngine,
        QualitySignals,
    )

    engine = QualitySignalEngine()
    signals = engine.compute_quality_signals(
        item_id="km_123",
        raw_confidence=0.85,
        contributors=["claude", "gpt-4"],
        contributor_ratings={"claude": rating1, "gpt-4": rating2},
    )
    print(f"Calibrated: {signals.calibrated_confidence:.2f}")
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """Dimensions of quality assessment."""

    CONFIDENCE = "confidence"
    """Raw confidence level."""

    CALIBRATION = "calibration"
    """How well-calibrated are contributors?"""

    FRESHNESS = "freshness"
    """Age-based quality decay."""

    SOURCE_RELIABILITY = "source_reliability"
    """Historical source accuracy."""

    CONSENSUS = "consensus"
    """Multi-contributor agreement."""

    VALIDATION = "validation"
    """Validation feedback history."""


class OverconfidenceLevel(Enum):
    """Levels of overconfidence severity."""

    NONE = "none"
    """No overconfidence detected."""

    MILD = "mild"
    """Slight overconfidence (ECE < 0.1)."""

    MODERATE = "moderate"
    """Moderate overconfidence (0.1 <= ECE < 0.2)."""

    SEVERE = "severe"
    """Severe overconfidence (ECE >= 0.2)."""


class QualityTier(Enum):
    """Quality tiers for knowledge items."""

    EXCELLENT = "excellent"
    """High quality, well-calibrated (score >= 0.9)."""

    GOOD = "good"
    """Above average quality (0.7 <= score < 0.9)."""

    ACCEPTABLE = "acceptable"
    """Acceptable quality (0.5 <= score < 0.7)."""

    POOR = "poor"
    """Below threshold, needs review (0.3 <= score < 0.5)."""

    UNRELIABLE = "unreliable"
    """Unreliable, should not be trusted (score < 0.3)."""


@dataclass
class CalibrationMetrics:
    """Calibration metrics for quality assessment."""

    brier_score: float = 0.0
    """Brier score (0-1, lower is better)."""

    expected_calibration_error: float = 0.0
    """ECE (0-1, lower is better)."""

    calibration_accuracy: float = 0.0
    """Fraction of correct predictions."""

    prediction_count: int = 0
    """Total predictions made."""

    overconfidence_ratio: float = 0.0
    """Ratio of overconfident predictions."""

    underconfidence_ratio: float = 0.0
    """Ratio of underconfident predictions."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "brier_score": round(self.brier_score, 4),
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "calibration_accuracy": round(self.calibration_accuracy, 4),
            "prediction_count": self.prediction_count,
            "overconfidence_ratio": round(self.overconfidence_ratio, 4),
            "underconfidence_ratio": round(self.underconfidence_ratio, 4),
        }


@dataclass
class SourceReliability:
    """Reliability metrics for a knowledge source."""

    source_id: str
    """Identifier for the source."""

    accuracy_score: float = 0.0
    """Historical accuracy (0-1)."""

    validation_count: int = 0
    """Number of validations."""

    correct_count: int = 0
    """Number of correct validations."""

    last_validated: Optional[datetime] = None
    """When last validated."""

    confidence_calibration: float = 1.0
    """Confidence scaling factor based on calibration."""

    domain_scores: Dict[str, float] = field(default_factory=dict)
    """Domain-specific reliability scores."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "accuracy_score": round(self.accuracy_score, 4),
            "validation_count": self.validation_count,
            "correct_count": self.correct_count,
            "last_validated": (self.last_validated.isoformat() if self.last_validated else None),
            "confidence_calibration": round(self.confidence_calibration, 4),
            "domain_scores": {k: round(v, 4) for k, v in self.domain_scores.items()},
        }


@dataclass
class QualitySignals:
    """Quality signals for a knowledge item."""

    item_id: str
    """ID of the knowledge item."""

    raw_confidence: float = 0.0
    """Original stated confidence."""

    calibrated_confidence: float = 0.0
    """Confidence adjusted by contributor calibration."""

    confidence_adjustment: float = 0.0
    """Difference between calibrated and raw confidence."""

    calibration_quality_index: float = 0.0
    """How well-calibrated are contributors? (0-1, higher is better)."""

    overconfidence_level: OverconfidenceLevel = OverconfidenceLevel.NONE
    """Severity of overconfidence in contributors."""

    overconfidence_flag: bool = False
    """Quick flag for overconfidence detection."""

    source_reliability: float = 0.0
    """Combined reliability of sources (0-1)."""

    contributor_weights: Dict[str, float] = field(default_factory=dict)
    """Weights assigned to each contributor."""

    quality_tier: QualityTier = QualityTier.ACCEPTABLE
    """Overall quality tier."""

    composite_quality_score: float = 0.0
    """Weighted combination of all quality dimensions (0-1)."""

    dimension_scores: Dict[str, float] = field(default_factory=dict)
    """Individual scores for each quality dimension."""

    warnings: List[str] = field(default_factory=list)
    """Quality warnings for this item."""

    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When signals were computed."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional quality metadata."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_id": self.item_id,
            "raw_confidence": round(self.raw_confidence, 4),
            "calibrated_confidence": round(self.calibrated_confidence, 4),
            "confidence_adjustment": round(self.confidence_adjustment, 4),
            "calibration_quality_index": round(self.calibration_quality_index, 4),
            "overconfidence_level": self.overconfidence_level.value,
            "overconfidence_flag": self.overconfidence_flag,
            "source_reliability": round(self.source_reliability, 4),
            "contributor_weights": {k: round(v, 4) for k, v in self.contributor_weights.items()},
            "quality_tier": self.quality_tier.value,
            "composite_quality_score": round(self.composite_quality_score, 4),
            "dimension_scores": {k: round(v, 4) for k, v in self.dimension_scores.items()},
            "warnings": self.warnings,
            "computed_at": self.computed_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ContributorCalibration:
    """Calibration data for a single contributor."""

    contributor_id: str
    """Identifier for the contributor."""

    calibration_accuracy: float = 0.0
    """Historical prediction accuracy (0-1)."""

    brier_score: float = 0.5
    """Brier score (lower is better)."""

    total_predictions: int = 0
    """Total number of predictions."""

    overconfidence_detected: bool = False
    """Whether contributor shows overconfidence."""

    reliability_weight: float = 1.0
    """Weight for this contributor."""


@dataclass
class QualityEngineConfig:
    """Configuration for the quality signal engine."""

    # ECE thresholds for overconfidence detection
    ece_mild_threshold: float = 0.05
    """ECE threshold for mild overconfidence."""

    ece_moderate_threshold: float = 0.10
    """ECE threshold for moderate overconfidence."""

    ece_severe_threshold: float = 0.20
    """ECE threshold for severe overconfidence."""

    # Calibration weighting
    min_predictions_for_weight: int = 5
    """Minimum predictions before weighting by calibration."""

    default_calibration_weight: float = 0.5
    """Default weight for uncalibrated contributors."""

    # Quality dimension weights
    dimension_weights: Dict[str, float] = field(
        default_factory=lambda: {
            QualityDimension.CONFIDENCE.value: 0.25,
            QualityDimension.CALIBRATION.value: 0.25,
            QualityDimension.SOURCE_RELIABILITY.value: 0.20,
            QualityDimension.CONSENSUS.value: 0.15,
            QualityDimension.FRESHNESS.value: 0.10,
            QualityDimension.VALIDATION.value: 0.05,
        }
    )

    # Quality tier thresholds
    excellent_threshold: float = 0.90
    good_threshold: float = 0.70
    acceptable_threshold: float = 0.50
    poor_threshold: float = 0.30

    # Source reliability
    min_validations_for_reliability: int = 3
    """Minimum validations before computing source reliability."""

    reliability_decay_factor: float = 0.95
    """Decay factor for older validations."""


class QualitySignalEngine:
    """Engine for computing calibration-driven quality signals.

    This engine computes quality metrics for knowledge items based on
    contributor calibration, source reliability, and validation history.
    """

    def __init__(self, config: Optional[QualityEngineConfig] = None):
        """Initialize the quality signal engine.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or QualityEngineConfig()
        self._source_reliability_cache: Dict[str, SourceReliability] = {}
        self._contributor_cache: Dict[str, ContributorCalibration] = {}

    def compute_quality_signals(
        self,
        item_id: str,
        raw_confidence: float,
        contributors: List[str],
        contributor_ratings: Optional[Dict[str, Any]] = None,
        sources: Optional[List[str]] = None,
        validation_history: Optional[Dict[str, List[bool]]] = None,
        domain: Optional[str] = None,
        created_at: Optional[datetime] = None,
    ) -> QualitySignals:
        """Compute quality signals for a knowledge item.

        Args:
            item_id: ID of the knowledge item.
            raw_confidence: Original stated confidence (0-1).
            contributors: List of contributor agent names.
            contributor_ratings: Optional dict of agent_name -> AgentRating.
            sources: Optional list of source identifiers.
            validation_history: Optional dict of source -> list of validation outcomes.
            domain: Optional domain for domain-specific calibration.
            created_at: Optional creation time for freshness calculation.

        Returns:
            QualitySignals with computed quality metrics.
        """
        signals = QualitySignals(
            item_id=item_id,
            raw_confidence=raw_confidence,
        )

        # Extract contributor calibrations
        contributor_calibrations = self._extract_contributor_calibrations(
            contributors, contributor_ratings, domain
        )

        # Compute calibrated confidence
        signals.calibrated_confidence = self._compute_calibrated_confidence(
            raw_confidence, contributor_calibrations
        )
        signals.confidence_adjustment = signals.calibrated_confidence - signals.raw_confidence

        # Store contributor weights
        signals.contributor_weights = {
            cc.contributor_id: cc.reliability_weight for cc in contributor_calibrations
        }

        # Compute calibration quality index
        signals.calibration_quality_index = self._compute_calibration_quality_index(
            contributor_calibrations
        )

        # Detect overconfidence
        overconfidence = self._detect_overconfidence(contributor_calibrations)
        signals.overconfidence_level = overconfidence
        signals.overconfidence_flag = overconfidence != OverconfidenceLevel.NONE

        # Compute source reliability
        if sources and validation_history:
            signals.source_reliability = self._compute_source_reliability(
                sources, validation_history
            )

        # Compute dimension scores
        signals.dimension_scores = self._compute_dimension_scores(
            signals=signals,
            contributor_calibrations=contributor_calibrations,
            created_at=created_at,
        )

        # Compute composite quality score
        signals.composite_quality_score = self._compute_composite_score(signals.dimension_scores)

        # Determine quality tier
        signals.quality_tier = self._determine_quality_tier(signals.composite_quality_score)

        # Generate warnings
        signals.warnings = self._generate_warnings(signals, contributor_calibrations)

        return signals

    def _extract_contributor_calibrations(
        self,
        contributors: List[str],
        contributor_ratings: Optional[Dict[str, Any]],
        domain: Optional[str],
    ) -> List[ContributorCalibration]:
        """Extract calibration data from contributor ratings.

        Args:
            contributors: List of contributor names.
            contributor_ratings: Optional ratings dict.
            domain: Optional domain for domain-specific calibration.

        Returns:
            List of ContributorCalibration objects.
        """
        calibrations = []

        for contributor in contributors:
            cc = ContributorCalibration(contributor_id=contributor)

            if contributor_ratings and contributor in contributor_ratings:
                rating = contributor_ratings[contributor]

                # Extract calibration fields (handle both dict and dataclass)
                if hasattr(rating, "calibration_accuracy"):
                    cc.calibration_accuracy = rating.calibration_accuracy
                    cc.total_predictions = getattr(rating, "calibration_total", 0)
                    brier_sum = getattr(rating, "calibration_brier_sum", 0.0)
                    if cc.total_predictions > 0:
                        cc.brier_score = brier_sum / cc.total_predictions
                elif isinstance(rating, dict):
                    cc.calibration_accuracy = rating.get("calibration_accuracy", 0.0)
                    cc.total_predictions = rating.get("calibration_total", 0)
                    brier_sum = rating.get("calibration_brier_sum", 0.0)
                    if cc.total_predictions > 0:
                        cc.brier_score = brier_sum / cc.total_predictions

                # Compute reliability weight based on calibration
                cc.reliability_weight = self._compute_contributor_weight(cc)

                # Detect overconfidence for this contributor
                cc.overconfidence_detected = cc.brier_score > 0.25  # High Brier = poor calibration

            else:
                # Unknown contributor, use defaults
                cc.reliability_weight = self.config.default_calibration_weight

            calibrations.append(cc)
            self._contributor_cache[contributor] = cc

        return calibrations

    def _compute_contributor_weight(self, cc: ContributorCalibration) -> float:
        """Compute weight for a contributor based on calibration.

        Args:
            cc: Contributor calibration data.

        Returns:
            Weight for this contributor (0-1).
        """
        if cc.total_predictions < self.config.min_predictions_for_weight:
            return self.config.default_calibration_weight

        # Weight based on calibration accuracy and Brier score
        accuracy_factor = cc.calibration_accuracy
        brier_factor = 1.0 - min(1.0, cc.brier_score)  # Lower Brier is better

        # Combine with more weight on Brier (it's a proper scoring rule)
        weight = 0.4 * accuracy_factor + 0.6 * brier_factor

        # Scale to [0.2, 1.0] range (never fully discount a contributor)
        return 0.2 + 0.8 * weight

    def _compute_calibrated_confidence(
        self,
        raw_confidence: float,
        contributor_calibrations: List[ContributorCalibration],
    ) -> float:
        """Compute calibrated confidence from contributor data.

        The calibrated confidence adjusts the raw confidence based on
        how well-calibrated the contributors historically are.

        Args:
            raw_confidence: Original confidence value.
            contributor_calibrations: Calibration data for contributors.

        Returns:
            Calibrated confidence (0-1).
        """
        if not contributor_calibrations:
            return raw_confidence

        # Compute weighted average calibration accuracy
        total_weight = 0.0
        weighted_adjustment = 0.0

        for cc in contributor_calibrations:
            weight = cc.reliability_weight

            # Adjustment factor based on Brier score
            # If Brier > 0.25, contributor tends to be overconfident
            if cc.total_predictions >= self.config.min_predictions_for_weight:
                # Scale raw confidence by calibration quality
                calibration_factor = 1.0 - cc.brier_score  # 0 to 1
                adjustment = raw_confidence * calibration_factor
            else:
                # Not enough data, use raw confidence with slight discount
                adjustment = raw_confidence * 0.9

            weighted_adjustment += weight * adjustment
            total_weight += weight

        if total_weight == 0:
            return raw_confidence

        calibrated = weighted_adjustment / total_weight

        # Clamp to valid range
        return max(0.0, min(1.0, calibrated))

    def _compute_calibration_quality_index(
        self,
        contributor_calibrations: List[ContributorCalibration],
    ) -> float:
        """Compute overall calibration quality index.

        This measures how well-calibrated the contributors are as a group.
        Higher is better.

        Args:
            contributor_calibrations: Calibration data for contributors.

        Returns:
            Calibration quality index (0-1).
        """
        if not contributor_calibrations:
            return 0.5  # Unknown

        # Average Brier score (lower is better, so invert)
        total_brier = 0.0
        total_accuracy = 0.0
        count = 0

        for cc in contributor_calibrations:
            if cc.total_predictions >= self.config.min_predictions_for_weight:
                total_brier += cc.brier_score
                total_accuracy += cc.calibration_accuracy
                count += 1

        if count == 0:
            return 0.5  # Unknown

        avg_brier = total_brier / count
        avg_accuracy = total_accuracy / count

        # Quality index: combine inverted Brier and accuracy
        brier_quality = 1.0 - min(1.0, avg_brier)
        quality_index = 0.6 * brier_quality + 0.4 * avg_accuracy

        return quality_index

    def _detect_overconfidence(
        self,
        contributor_calibrations: List[ContributorCalibration],
    ) -> OverconfidenceLevel:
        """Detect overconfidence in contributors.

        Args:
            contributor_calibrations: Calibration data for contributors.

        Returns:
            Overconfidence level.
        """
        if not contributor_calibrations:
            return OverconfidenceLevel.NONE

        # Compute average "overconfidence" from Brier scores
        # High Brier with high stated confidence = overconfidence
        overconfidence_scores = []

        for cc in contributor_calibrations:
            if cc.total_predictions >= self.config.min_predictions_for_weight:
                # Use Brier score as proxy for miscalibration
                overconfidence_scores.append(cc.brier_score)

        if not overconfidence_scores:
            return OverconfidenceLevel.NONE

        avg_brier = sum(overconfidence_scores) / len(overconfidence_scores)

        # Map Brier to overconfidence level using ECE-like thresholds
        # (Brier is similar to ECE in detecting miscalibration)
        if avg_brier >= self.config.ece_severe_threshold:
            return OverconfidenceLevel.SEVERE
        elif avg_brier >= self.config.ece_moderate_threshold:
            return OverconfidenceLevel.MODERATE
        elif avg_brier >= self.config.ece_mild_threshold:
            return OverconfidenceLevel.MILD
        else:
            return OverconfidenceLevel.NONE

    def _compute_source_reliability(
        self,
        sources: List[str],
        validation_history: Dict[str, List[bool]],
    ) -> float:
        """Compute combined source reliability.

        Args:
            sources: List of source identifiers.
            validation_history: Dict of source -> list of validation outcomes.

        Returns:
            Combined reliability score (0-1).
        """
        if not sources:
            return 0.5  # Unknown

        reliabilities = []
        decay_factor = self.config.reliability_decay_factor

        for source in sources:
            if source in validation_history:
                outcomes = validation_history[source]

                if len(outcomes) >= self.config.min_validations_for_reliability:
                    # Apply time decay (more recent validations weight more)
                    weighted_sum = 0.0
                    weight_total = 0.0
                    weight = 1.0

                    # Assume outcomes are ordered newest first
                    for outcome in outcomes:
                        weighted_sum += weight * (1.0 if outcome else 0.0)
                        weight_total += weight
                        weight *= decay_factor

                    reliability = weighted_sum / weight_total if weight_total > 0 else 0.5

                    # Cache for future use
                    self._source_reliability_cache[source] = SourceReliability(
                        source_id=source,
                        accuracy_score=reliability,
                        validation_count=len(outcomes),
                        correct_count=sum(1 for o in outcomes if o),
                        last_validated=datetime.now(timezone.utc),
                    )

                    reliabilities.append(reliability)
                else:
                    reliabilities.append(0.5)  # Not enough data
            else:
                reliabilities.append(0.5)  # Unknown source

        return sum(reliabilities) / len(reliabilities) if reliabilities else 0.5

    def _compute_dimension_scores(
        self,
        signals: QualitySignals,
        contributor_calibrations: List[ContributorCalibration],
        created_at: Optional[datetime],
    ) -> Dict[str, float]:
        """Compute individual dimension scores.

        Args:
            signals: Partially computed quality signals.
            contributor_calibrations: Calibration data.
            created_at: Item creation time for freshness.

        Returns:
            Dict of dimension -> score.
        """
        scores = {}

        # Confidence dimension
        scores[QualityDimension.CONFIDENCE.value] = signals.calibrated_confidence

        # Calibration dimension
        scores[QualityDimension.CALIBRATION.value] = signals.calibration_quality_index

        # Source reliability dimension
        scores[QualityDimension.SOURCE_RELIABILITY.value] = signals.source_reliability or 0.5

        # Consensus dimension (based on contributor agreement)
        if len(contributor_calibrations) > 1:
            # Use variance in weights as proxy for disagreement
            weights = [cc.reliability_weight for cc in contributor_calibrations]
            mean_weight = sum(weights) / len(weights)
            variance = sum((w - mean_weight) ** 2 for w in weights) / len(weights)
            # Low variance = high consensus
            scores[QualityDimension.CONSENSUS.value] = max(0, 1.0 - variance * 2)
        else:
            scores[QualityDimension.CONSENSUS.value] = 0.7  # Single contributor

        # Freshness dimension
        if created_at:
            age_days = (datetime.now(timezone.utc) - created_at).total_seconds() / 86400
            # Decay over 365 days
            freshness = math.exp(-age_days / 365)
            scores[QualityDimension.FRESHNESS.value] = max(0, min(1, freshness))
        else:
            scores[QualityDimension.FRESHNESS.value] = 0.8  # Unknown age, assume recent

        # Validation dimension (placeholder - would need validation data)
        scores[QualityDimension.VALIDATION.value] = 0.5

        return scores

    def _compute_composite_score(
        self,
        dimension_scores: Dict[str, float],
    ) -> float:
        """Compute weighted composite quality score.

        Args:
            dimension_scores: Individual dimension scores.

        Returns:
            Composite score (0-1).
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for dimension, score in dimension_scores.items():
            weight = self.config.dimension_weights.get(dimension, 0.1)
            weighted_sum += weight * score
            total_weight += weight

        if total_weight == 0:
            return 0.5

        return weighted_sum / total_weight

    def _determine_quality_tier(self, composite_score: float) -> QualityTier:
        """Determine quality tier from composite score.

        Args:
            composite_score: The composite quality score.

        Returns:
            Appropriate QualityTier.
        """
        if composite_score >= self.config.excellent_threshold:
            return QualityTier.EXCELLENT
        elif composite_score >= self.config.good_threshold:
            return QualityTier.GOOD
        elif composite_score >= self.config.acceptable_threshold:
            return QualityTier.ACCEPTABLE
        elif composite_score >= self.config.poor_threshold:
            return QualityTier.POOR
        else:
            return QualityTier.UNRELIABLE

    def _generate_warnings(
        self,
        signals: QualitySignals,
        contributor_calibrations: List[ContributorCalibration],
    ) -> List[str]:
        """Generate quality warnings.

        Args:
            signals: Computed quality signals.
            contributor_calibrations: Calibration data.

        Returns:
            List of warning messages.
        """
        warnings = []

        # Overconfidence warning
        if signals.overconfidence_level == OverconfidenceLevel.SEVERE:
            warnings.append(
                "Severe overconfidence detected in contributors. "
                "Confidence has been significantly adjusted."
            )
        elif signals.overconfidence_level == OverconfidenceLevel.MODERATE:
            warnings.append("Moderate overconfidence detected. Consider additional validation.")

        # Large adjustment warning
        if abs(signals.confidence_adjustment) > 0.2:
            direction = "increased" if signals.confidence_adjustment > 0 else "decreased"
            warnings.append(
                f"Confidence {direction} by {abs(signals.confidence_adjustment):.0%} "
                "based on contributor calibration history."
            )

        # Low calibration quality warning
        if signals.calibration_quality_index < 0.4:
            warnings.append(
                "Contributors have poor calibration history. Quality signals may be unreliable."
            )

        # Low source reliability warning
        if signals.source_reliability < 0.4:
            warnings.append(
                "Sources have low historical reliability. Consider independent verification."
            )

        # Single contributor warning
        if len(contributor_calibrations) == 1:
            warnings.append("Only one contributor. Multi-agent validation recommended.")

        # Uncalibrated contributor warning
        uncalibrated = sum(
            1
            for cc in contributor_calibrations
            if cc.total_predictions < self.config.min_predictions_for_weight
        )
        if uncalibrated > 0:
            warnings.append(f"{uncalibrated} contributor(s) have insufficient calibration history.")

        return warnings

    def compute_expected_calibration_error(
        self,
        predictions: List[Tuple[float, bool]],
        num_buckets: int = 10,
    ) -> float:
        """Compute Expected Calibration Error (ECE).

        ECE measures the average gap between predicted confidence
        and actual accuracy across confidence buckets.

        Args:
            predictions: List of (confidence, correct) tuples.
            num_buckets: Number of confidence buckets.

        Returns:
            ECE value (0-1, lower is better).
        """
        if not predictions:
            return 0.0

        bucket_size = 1.0 / num_buckets
        buckets: Dict[int, List[Tuple[float, bool]]] = defaultdict(list)

        for confidence, correct in predictions:
            bucket_idx = min(int(confidence / bucket_size), num_buckets - 1)
            buckets[bucket_idx].append((confidence, correct))

        ece = 0.0
        total = len(predictions)

        for bucket_idx, bucket_preds in buckets.items():
            if not bucket_preds:
                continue

            bucket_size_count = len(bucket_preds)
            avg_confidence = sum(c for c, _ in bucket_preds) / bucket_size_count
            accuracy = sum(1 for _, correct in bucket_preds if correct) / bucket_size_count

            ece += (bucket_size_count / total) * abs(avg_confidence - accuracy)

        return ece

    def detect_contributor_overconfidence(
        self,
        agent_name: str,
        predictions: List[Tuple[float, bool]],
        threshold: float = 0.1,
    ) -> bool:
        """Detect if a contributor is consistently overconfident.

        Args:
            agent_name: Name of the contributor.
            predictions: List of (confidence, correct) tuples.
            threshold: ECE threshold for overconfidence.

        Returns:
            True if contributor is overconfident.
        """
        if len(predictions) < self.config.min_predictions_for_weight:
            return False

        ece = self.compute_expected_calibration_error(predictions)

        # Check direction of miscalibration
        # Overconfidence = high confidence but low accuracy
        total_confidence = sum(c for c, _ in predictions)
        total_correct = sum(1 for _, correct in predictions if correct)

        avg_confidence = total_confidence / len(predictions)
        accuracy = total_correct / len(predictions)

        # Overconfident if ECE is high AND confidence > accuracy
        return ece > threshold and avg_confidence > accuracy

    def batch_compute_signals(
        self,
        items: List[Dict[str, Any]],
        contributor_ratings: Optional[Dict[str, Any]] = None,
    ) -> List[QualitySignals]:
        """Compute quality signals for multiple items.

        Args:
            items: List of item dicts with keys: item_id, confidence, contributors.
            contributor_ratings: Optional shared ratings dict.

        Returns:
            List of QualitySignals for each item.
        """
        results = []

        for item in items:
            signals = self.compute_quality_signals(
                item_id=item.get("item_id", "unknown"),
                raw_confidence=item.get("confidence", 0.5),
                contributors=item.get("contributors", []),
                contributor_ratings=contributor_ratings,
                sources=item.get("sources"),
                validation_history=item.get("validation_history"),
                domain=item.get("domain"),
                created_at=item.get("created_at"),
            )
            results.append(signals)

        return results

    def get_quality_summary(
        self,
        signals_list: List[QualitySignals],
    ) -> Dict[str, Any]:
        """Generate summary statistics for multiple quality signals.

        Args:
            signals_list: List of QualitySignals.

        Returns:
            Summary statistics dict.
        """
        if not signals_list:
            return {"count": 0}

        # Count by tier
        tier_counts: Dict[str, int] = defaultdict(int)
        for s in signals_list:
            tier_counts[s.quality_tier.value] += 1

        # Average scores
        avg_raw = sum(s.raw_confidence for s in signals_list) / len(signals_list)
        avg_calibrated = sum(s.calibrated_confidence for s in signals_list) / len(signals_list)
        avg_composite = sum(s.composite_quality_score for s in signals_list) / len(signals_list)
        avg_calibration_index = sum(s.calibration_quality_index for s in signals_list) / len(
            signals_list
        )

        # Overconfidence counts
        overconfident_count = sum(1 for s in signals_list if s.overconfidence_flag)

        # Warning counts
        total_warnings = sum(len(s.warnings) for s in signals_list)

        return {
            "count": len(signals_list),
            "tier_distribution": dict(tier_counts),
            "avg_raw_confidence": round(avg_raw, 4),
            "avg_calibrated_confidence": round(avg_calibrated, 4),
            "avg_composite_score": round(avg_composite, 4),
            "avg_calibration_index": round(avg_calibration_index, 4),
            "overconfident_items": overconfident_count,
            "overconfident_ratio": round(overconfident_count / len(signals_list), 4),
            "total_warnings": total_warnings,
            "avg_warnings_per_item": round(total_warnings / len(signals_list), 2),
        }


# Singleton instance
_quality_signal_engine: Optional[QualitySignalEngine] = None


def get_quality_signal_engine(
    config: Optional[QualityEngineConfig] = None,
) -> QualitySignalEngine:
    """Get or create the singleton quality signal engine.

    Args:
        config: Optional configuration (only used on first call).

    Returns:
        QualitySignalEngine instance.
    """
    global _quality_signal_engine
    if _quality_signal_engine is None:
        _quality_signal_engine = QualitySignalEngine(config)
    return _quality_signal_engine


__all__ = [
    # Enums
    "QualityDimension",
    "OverconfidenceLevel",
    "QualityTier",
    # Dataclasses
    "CalibrationMetrics",
    "SourceReliability",
    "QualitySignals",
    "ContributorCalibration",
    "QualityEngineConfig",
    # Engine
    "QualitySignalEngine",
    "get_quality_signal_engine",
]
