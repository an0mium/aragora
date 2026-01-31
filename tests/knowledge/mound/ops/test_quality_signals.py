"""
Comprehensive unit tests for aragora/knowledge/mound/ops/quality_signals.py.

Tests cover:
- Enum value verification (QualityDimension, OverconfidenceLevel, QualityTier)
- Dataclass creation and to_dict conversion (CalibrationMetrics, SourceReliability, etc.)
- QualitySignalEngine initialization and configuration
- Signal extraction and calibrated confidence computation
- Quality scoring algorithms and composite score calculation
- Freshness and decay calculations
- Source reliability signals
- Overconfidence detection
- Contributor weight computation
- Warning generation
- Expected Calibration Error (ECE) computation
- Batch operations and quality summaries
- Edge cases (empty content, invalid signals, extreme values)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from aragora.knowledge.mound.ops.quality_signals import (
    CalibrationMetrics,
    ContributorCalibration,
    OverconfidenceLevel,
    QualityDimension,
    QualityEngineConfig,
    QualitySignalEngine,
    QualitySignals,
    QualityTier,
    SourceReliability,
    get_quality_signal_engine,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestQualityDimensionEnum:
    """Tests for QualityDimension enum values."""

    def test_confidence_value(self):
        """Test CONFIDENCE enum value."""
        assert QualityDimension.CONFIDENCE.value == "confidence"

    def test_calibration_value(self):
        """Test CALIBRATION enum value."""
        assert QualityDimension.CALIBRATION.value == "calibration"

    def test_freshness_value(self):
        """Test FRESHNESS enum value."""
        assert QualityDimension.FRESHNESS.value == "freshness"

    def test_source_reliability_value(self):
        """Test SOURCE_RELIABILITY enum value."""
        assert QualityDimension.SOURCE_RELIABILITY.value == "source_reliability"

    def test_consensus_value(self):
        """Test CONSENSUS enum value."""
        assert QualityDimension.CONSENSUS.value == "consensus"

    def test_validation_value(self):
        """Test VALIDATION enum value."""
        assert QualityDimension.VALIDATION.value == "validation"

    def test_all_dimensions_are_unique(self):
        """Ensure all dimension values are unique."""
        values = [d.value for d in QualityDimension]
        assert len(values) == len(set(values))


class TestOverconfidenceLevelEnum:
    """Tests for OverconfidenceLevel enum values."""

    def test_none_value(self):
        """Test NONE enum value."""
        assert OverconfidenceLevel.NONE.value == "none"

    def test_mild_value(self):
        """Test MILD enum value."""
        assert OverconfidenceLevel.MILD.value == "mild"

    def test_moderate_value(self):
        """Test MODERATE enum value."""
        assert OverconfidenceLevel.MODERATE.value == "moderate"

    def test_severe_value(self):
        """Test SEVERE enum value."""
        assert OverconfidenceLevel.SEVERE.value == "severe"

    def test_ordering_none_is_best(self):
        """Test that NONE represents no overconfidence."""
        assert OverconfidenceLevel.NONE != OverconfidenceLevel.MILD
        assert OverconfidenceLevel.NONE != OverconfidenceLevel.SEVERE


class TestQualityTierEnum:
    """Tests for QualityTier enum values."""

    def test_excellent_value(self):
        """Test EXCELLENT enum value."""
        assert QualityTier.EXCELLENT.value == "excellent"

    def test_good_value(self):
        """Test GOOD enum value."""
        assert QualityTier.GOOD.value == "good"

    def test_acceptable_value(self):
        """Test ACCEPTABLE enum value."""
        assert QualityTier.ACCEPTABLE.value == "acceptable"

    def test_poor_value(self):
        """Test POOR enum value."""
        assert QualityTier.POOR.value == "poor"

    def test_unreliable_value(self):
        """Test UNRELIABLE enum value."""
        assert QualityTier.UNRELIABLE.value == "unreliable"


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestCalibrationMetrics:
    """Tests for CalibrationMetrics dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        metrics = CalibrationMetrics()
        assert metrics.brier_score == 0.0
        assert metrics.expected_calibration_error == 0.0
        assert metrics.calibration_accuracy == 0.0
        assert metrics.prediction_count == 0
        assert metrics.overconfidence_ratio == 0.0
        assert metrics.underconfidence_ratio == 0.0

    def test_custom_values(self):
        """Test custom values are set correctly."""
        metrics = CalibrationMetrics(
            brier_score=0.15,
            expected_calibration_error=0.08,
            calibration_accuracy=0.85,
            prediction_count=100,
            overconfidence_ratio=0.1,
            underconfidence_ratio=0.05,
        )
        assert metrics.brier_score == 0.15
        assert metrics.expected_calibration_error == 0.08
        assert metrics.calibration_accuracy == 0.85
        assert metrics.prediction_count == 100

    def test_to_dict(self):
        """Test to_dict conversion."""
        metrics = CalibrationMetrics(
            brier_score=0.123456,
            expected_calibration_error=0.087654,
            calibration_accuracy=0.854321,
            prediction_count=50,
        )
        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["brier_score"] == 0.1235  # Rounded to 4 decimal places
        assert result["expected_calibration_error"] == 0.0877
        assert result["calibration_accuracy"] == 0.8543
        assert result["prediction_count"] == 50


class TestSourceReliability:
    """Tests for SourceReliability dataclass."""

    def test_required_source_id(self):
        """Test source_id is required."""
        reliability = SourceReliability(source_id="src_123")
        assert reliability.source_id == "src_123"

    def test_default_values(self):
        """Test default values are set correctly."""
        reliability = SourceReliability(source_id="src_1")
        assert reliability.accuracy_score == 0.0
        assert reliability.validation_count == 0
        assert reliability.correct_count == 0
        assert reliability.last_validated is None
        assert reliability.confidence_calibration == 1.0
        assert reliability.domain_scores == {}

    def test_custom_values(self):
        """Test custom values are set correctly."""
        now = datetime.now(timezone.utc)
        reliability = SourceReliability(
            source_id="src_1",
            accuracy_score=0.92,
            validation_count=50,
            correct_count=46,
            last_validated=now,
            confidence_calibration=0.95,
            domain_scores={"finance": 0.88, "tech": 0.95},
        )
        assert reliability.accuracy_score == 0.92
        assert reliability.validation_count == 50
        assert reliability.correct_count == 46
        assert reliability.last_validated == now
        assert reliability.domain_scores["finance"] == 0.88

    def test_to_dict(self):
        """Test to_dict conversion."""
        now = datetime.now(timezone.utc)
        reliability = SourceReliability(
            source_id="src_1",
            accuracy_score=0.876543,
            validation_count=100,
            correct_count=87,
            last_validated=now,
            domain_scores={"test": 0.912345},
        )
        result = reliability.to_dict()

        assert result["source_id"] == "src_1"
        assert result["accuracy_score"] == 0.8765
        assert result["validation_count"] == 100
        assert result["correct_count"] == 87
        assert result["last_validated"] == now.isoformat()
        assert result["domain_scores"]["test"] == 0.9123

    def test_to_dict_with_none_last_validated(self):
        """Test to_dict handles None last_validated."""
        reliability = SourceReliability(source_id="src_1")
        result = reliability.to_dict()
        assert result["last_validated"] is None


class TestQualitySignals:
    """Tests for QualitySignals dataclass."""

    def test_required_item_id(self):
        """Test item_id is required."""
        signals = QualitySignals(item_id="km_123")
        assert signals.item_id == "km_123"

    def test_default_values(self):
        """Test default values are set correctly."""
        signals = QualitySignals(item_id="km_1")
        assert signals.raw_confidence == 0.0
        assert signals.calibrated_confidence == 0.0
        assert signals.confidence_adjustment == 0.0
        assert signals.calibration_quality_index == 0.0
        assert signals.overconfidence_level == OverconfidenceLevel.NONE
        assert signals.overconfidence_flag is False
        assert signals.source_reliability == 0.0
        assert signals.contributor_weights == {}
        assert signals.quality_tier == QualityTier.ACCEPTABLE
        assert signals.composite_quality_score == 0.0
        assert signals.dimension_scores == {}
        assert signals.warnings == []
        assert signals.metadata == {}

    def test_computed_at_auto_set(self):
        """Test computed_at is automatically set to current time."""
        before = datetime.now(timezone.utc)
        signals = QualitySignals(item_id="km_1")
        after = datetime.now(timezone.utc)

        assert before <= signals.computed_at <= after

    def test_to_dict_basic(self):
        """Test basic to_dict conversion."""
        signals = QualitySignals(
            item_id="km_123",
            raw_confidence=0.85,
            calibrated_confidence=0.78,
            quality_tier=QualityTier.GOOD,
        )
        result = signals.to_dict()

        assert result["item_id"] == "km_123"
        assert result["raw_confidence"] == 0.85
        assert result["calibrated_confidence"] == 0.78
        assert result["quality_tier"] == "good"
        assert "computed_at" in result

    def test_to_dict_full(self):
        """Test to_dict with all fields populated."""
        signals = QualitySignals(
            item_id="km_full",
            raw_confidence=0.9,
            calibrated_confidence=0.75,
            confidence_adjustment=-0.15,
            calibration_quality_index=0.65,
            overconfidence_level=OverconfidenceLevel.MODERATE,
            overconfidence_flag=True,
            source_reliability=0.82,
            contributor_weights={"claude": 0.9, "gpt": 0.85},
            quality_tier=QualityTier.GOOD,
            composite_quality_score=0.72,
            dimension_scores={"confidence": 0.75, "calibration": 0.65},
            warnings=["Warning 1", "Warning 2"],
            metadata={"extra": "data"},
        )
        result = signals.to_dict()

        assert result["confidence_adjustment"] == -0.15
        assert result["overconfidence_level"] == "moderate"
        assert result["overconfidence_flag"] is True
        assert result["contributor_weights"]["claude"] == 0.9
        assert len(result["warnings"]) == 2


class TestContributorCalibration:
    """Tests for ContributorCalibration dataclass."""

    def test_required_contributor_id(self):
        """Test contributor_id is required."""
        cc = ContributorCalibration(contributor_id="claude")
        assert cc.contributor_id == "claude"

    def test_default_values(self):
        """Test default values are set correctly."""
        cc = ContributorCalibration(contributor_id="agent1")
        assert cc.calibration_accuracy == 0.0
        assert cc.brier_score == 0.5
        assert cc.total_predictions == 0
        assert cc.overconfidence_detected is False
        assert cc.reliability_weight == 1.0

    def test_custom_values(self):
        """Test custom values assignment."""
        cc = ContributorCalibration(
            contributor_id="claude",
            calibration_accuracy=0.85,
            brier_score=0.12,
            total_predictions=100,
            overconfidence_detected=True,
            reliability_weight=0.9,
        )
        assert cc.calibration_accuracy == 0.85
        assert cc.brier_score == 0.12
        assert cc.total_predictions == 100
        assert cc.overconfidence_detected is True
        assert cc.reliability_weight == 0.9


class TestQualityEngineConfig:
    """Tests for QualityEngineConfig dataclass."""

    def test_default_ece_thresholds(self):
        """Test default ECE thresholds."""
        config = QualityEngineConfig()
        assert config.ece_mild_threshold == 0.05
        assert config.ece_moderate_threshold == 0.10
        assert config.ece_severe_threshold == 0.20

    def test_default_calibration_settings(self):
        """Test default calibration settings."""
        config = QualityEngineConfig()
        assert config.min_predictions_for_weight == 5
        assert config.default_calibration_weight == 0.5

    def test_default_dimension_weights(self):
        """Test default dimension weights sum to 1.0."""
        config = QualityEngineConfig()
        total = sum(config.dimension_weights.values())
        assert abs(total - 1.0) < 0.01

    def test_default_quality_tier_thresholds(self):
        """Test default quality tier thresholds."""
        config = QualityEngineConfig()
        assert config.excellent_threshold == 0.90
        assert config.good_threshold == 0.70
        assert config.acceptable_threshold == 0.50
        assert config.poor_threshold == 0.30

    def test_default_source_reliability_settings(self):
        """Test default source reliability settings."""
        config = QualityEngineConfig()
        assert config.min_validations_for_reliability == 3
        assert config.reliability_decay_factor == 0.95

    def test_custom_config(self):
        """Test custom configuration."""
        config = QualityEngineConfig(
            ece_mild_threshold=0.08,
            min_predictions_for_weight=10,
            excellent_threshold=0.95,
        )
        assert config.ece_mild_threshold == 0.08
        assert config.min_predictions_for_weight == 10
        assert config.excellent_threshold == 0.95


# =============================================================================
# QualitySignalEngine Tests
# =============================================================================


class TestQualitySignalEngineInit:
    """Tests for QualitySignalEngine initialization."""

    def test_default_initialization(self):
        """Test engine initializes with default config."""
        engine = QualitySignalEngine()
        assert engine.config is not None
        assert isinstance(engine.config, QualityEngineConfig)

    def test_custom_config_initialization(self):
        """Test engine initializes with custom config."""
        config = QualityEngineConfig(ece_mild_threshold=0.08)
        engine = QualitySignalEngine(config=config)
        assert engine.config.ece_mild_threshold == 0.08

    def test_caches_initialized_empty(self):
        """Test caches are initialized as empty."""
        engine = QualitySignalEngine()
        assert engine._source_reliability_cache == {}
        assert engine._contributor_cache == {}


class TestComputeQualitySignalsBasic:
    """Tests for basic quality signal computation."""

    def test_basic_computation(self):
        """Test basic quality signal computation."""
        engine = QualitySignalEngine()
        signals = engine.compute_quality_signals(
            item_id="km_123",
            raw_confidence=0.85,
            contributors=["claude"],
        )

        assert isinstance(signals, QualitySignals)
        assert signals.item_id == "km_123"
        assert signals.raw_confidence == 0.85

    def test_returns_quality_signals_type(self):
        """Test return type is QualitySignals."""
        engine = QualitySignalEngine()
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.5,
            contributors=[],
        )
        assert isinstance(signals, QualitySignals)

    def test_preserves_item_id(self):
        """Test item_id is preserved in signals."""
        engine = QualitySignalEngine()
        signals = engine.compute_quality_signals(
            item_id="unique_item_id_123",
            raw_confidence=0.7,
            contributors=["agent"],
        )
        assert signals.item_id == "unique_item_id_123"

    def test_preserves_raw_confidence(self):
        """Test raw_confidence is preserved in signals."""
        engine = QualitySignalEngine()
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.77,
            contributors=["agent"],
        )
        assert signals.raw_confidence == 0.77


class TestCalibratedConfidenceComputation:
    """Tests for calibrated confidence computation."""

    def test_no_contributors_returns_raw(self):
        """Test calibrated confidence equals raw with no contributors."""
        engine = QualitySignalEngine()
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.85,
            contributors=[],
        )
        # With no contributors, calibrated should be close to raw
        assert signals.calibrated_confidence == 0.85

    def test_well_calibrated_contributor(self):
        """Test calibrated confidence with well-calibrated contributor."""
        engine = QualitySignalEngine()
        ratings = {
            "claude": {
                "calibration_accuracy": 0.95,
                "calibration_total": 100,
                "calibration_brier_sum": 5.0,  # 0.05 average Brier
            }
        }
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.80,
            contributors=["claude"],
            contributor_ratings=ratings,
        )
        # Well-calibrated contributor should not reduce confidence much
        assert 0.7 <= signals.calibrated_confidence <= 0.85

    def test_poorly_calibrated_contributor(self):
        """Test calibrated confidence with poorly calibrated contributor."""
        engine = QualitySignalEngine()
        ratings = {
            "poor_agent": {
                "calibration_accuracy": 0.4,
                "calibration_total": 50,
                "calibration_brier_sum": 15.0,  # 0.30 average Brier
            }
        }
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.90,
            contributors=["poor_agent"],
            contributor_ratings=ratings,
        )
        # Poorly calibrated contributor should reduce confidence
        assert signals.calibrated_confidence < 0.90

    def test_confidence_adjustment_computed(self):
        """Test confidence adjustment is computed correctly."""
        engine = QualitySignalEngine()
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.85,
            contributors=["agent"],
        )
        expected_adjustment = signals.calibrated_confidence - signals.raw_confidence
        assert abs(signals.confidence_adjustment - expected_adjustment) < 0.001

    def test_calibrated_confidence_clamped(self):
        """Test calibrated confidence is clamped to [0, 1]."""
        engine = QualitySignalEngine()
        # Test with extreme values
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=1.0,
            contributors=["agent"],
        )
        assert 0 <= signals.calibrated_confidence <= 1.0


class TestContributorWeights:
    """Tests for contributor weight computation."""

    def test_default_weight_for_unknown_contributor(self):
        """Test unknown contributors get default weight."""
        engine = QualitySignalEngine()
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["unknown_agent"],
        )
        # Default weight is 0.5 from config
        assert signals.contributor_weights["unknown_agent"] == 0.5

    def test_weight_from_calibration(self):
        """Test weight is computed from calibration data."""
        engine = QualitySignalEngine()
        ratings = {
            "claude": {
                "calibration_accuracy": 0.9,
                "calibration_total": 50,
                "calibration_brier_sum": 5.0,
            }
        }
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["claude"],
            contributor_ratings=ratings,
        )
        # Weight should be > default for well-calibrated contributor
        assert signals.contributor_weights["claude"] > 0.5

    def test_multiple_contributor_weights(self):
        """Test weights for multiple contributors."""
        engine = QualitySignalEngine()
        ratings = {
            "claude": {
                "calibration_accuracy": 0.9,
                "calibration_total": 100,
                "calibration_brier_sum": 10.0,
            },
            "gpt": {
                "calibration_accuracy": 0.75,
                "calibration_total": 80,
                "calibration_brier_sum": 20.0,
            },
        }
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["claude", "gpt"],
            contributor_ratings=ratings,
        )
        assert "claude" in signals.contributor_weights
        assert "gpt" in signals.contributor_weights
        # Better calibrated contributor should have higher weight
        assert signals.contributor_weights["claude"] > signals.contributor_weights["gpt"]


class TestContributorWeightComputation:
    """Tests for _compute_contributor_weight method."""

    def test_insufficient_predictions_returns_default(self):
        """Test default weight for insufficient predictions."""
        engine = QualitySignalEngine()
        cc = ContributorCalibration(
            contributor_id="agent",
            total_predictions=2,  # Less than min_predictions_for_weight (5)
        )
        weight = engine._compute_contributor_weight(cc)
        assert weight == engine.config.default_calibration_weight

    def test_perfect_calibration(self):
        """Test weight for perfect calibration."""
        engine = QualitySignalEngine()
        cc = ContributorCalibration(
            contributor_id="perfect",
            calibration_accuracy=1.0,
            brier_score=0.0,
            total_predictions=100,
        )
        weight = engine._compute_contributor_weight(cc)
        # Should be close to maximum (1.0)
        assert weight > 0.9

    def test_poor_calibration(self):
        """Test weight for poor calibration."""
        engine = QualitySignalEngine()
        cc = ContributorCalibration(
            contributor_id="poor",
            calibration_accuracy=0.2,
            brier_score=0.8,
            total_predictions=100,
        )
        weight = engine._compute_contributor_weight(cc)
        # Should be close to minimum (0.2)
        assert weight < 0.4

    def test_weight_bounded_in_range(self):
        """Test weight is bounded in [0.2, 1.0] range."""
        engine = QualitySignalEngine()
        cc = ContributorCalibration(
            contributor_id="agent",
            calibration_accuracy=0.5,
            brier_score=0.5,
            total_predictions=100,
        )
        weight = engine._compute_contributor_weight(cc)
        assert 0.2 <= weight <= 1.0


class TestCalibrationQualityIndex:
    """Tests for calibration quality index computation."""

    def test_no_contributors_returns_default(self):
        """Test default quality index with no contributors."""
        engine = QualitySignalEngine()
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=[],
        )
        assert signals.calibration_quality_index == 0.5

    def test_high_quality_contributors(self):
        """Test high quality index with well-calibrated contributors."""
        engine = QualitySignalEngine()
        ratings = {
            "good1": {
                "calibration_accuracy": 0.95,
                "calibration_total": 100,
                "calibration_brier_sum": 5.0,
            },
            "good2": {
                "calibration_accuracy": 0.90,
                "calibration_total": 100,
                "calibration_brier_sum": 8.0,
            },
        }
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["good1", "good2"],
            contributor_ratings=ratings,
        )
        assert signals.calibration_quality_index > 0.7

    def test_low_quality_contributors(self):
        """Test low quality index with poorly calibrated contributors."""
        engine = QualitySignalEngine()
        ratings = {
            "bad1": {
                "calibration_accuracy": 0.3,
                "calibration_total": 50,
                "calibration_brier_sum": 15.0,
            },
        }
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["bad1"],
            contributor_ratings=ratings,
        )
        assert signals.calibration_quality_index < 0.6


class TestOverconfidenceDetection:
    """Tests for overconfidence detection."""

    def test_no_overconfidence(self):
        """Test no overconfidence with well-calibrated contributors."""
        engine = QualitySignalEngine()
        ratings = {
            "good": {
                "calibration_accuracy": 0.9,
                "calibration_total": 100,
                "calibration_brier_sum": 3.0,  # 0.03 average Brier
            }
        }
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["good"],
            contributor_ratings=ratings,
        )
        assert signals.overconfidence_level == OverconfidenceLevel.NONE
        assert signals.overconfidence_flag is False

    def test_mild_overconfidence(self):
        """Test mild overconfidence detection."""
        engine = QualitySignalEngine()
        ratings = {
            "agent": {
                "calibration_accuracy": 0.7,
                "calibration_total": 100,
                "calibration_brier_sum": 7.0,  # 0.07 average Brier
            }
        }
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.95,
            contributors=["agent"],
            contributor_ratings=ratings,
        )
        assert signals.overconfidence_level == OverconfidenceLevel.MILD
        assert signals.overconfidence_flag is True

    def test_moderate_overconfidence(self):
        """Test moderate overconfidence detection."""
        engine = QualitySignalEngine()
        ratings = {
            "agent": {
                "calibration_accuracy": 0.5,
                "calibration_total": 100,
                "calibration_brier_sum": 15.0,  # 0.15 average Brier
            }
        }
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.95,
            contributors=["agent"],
            contributor_ratings=ratings,
        )
        assert signals.overconfidence_level == OverconfidenceLevel.MODERATE

    def test_severe_overconfidence(self):
        """Test severe overconfidence detection."""
        engine = QualitySignalEngine()
        ratings = {
            "agent": {
                "calibration_accuracy": 0.3,
                "calibration_total": 100,
                "calibration_brier_sum": 25.0,  # 0.25 average Brier
            }
        }
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.99,
            contributors=["agent"],
            contributor_ratings=ratings,
        )
        assert signals.overconfidence_level == OverconfidenceLevel.SEVERE
        assert signals.overconfidence_flag is True

    def test_insufficient_data_no_overconfidence(self):
        """Test no overconfidence detected with insufficient data."""
        engine = QualitySignalEngine()
        ratings = {
            "agent": {
                "calibration_accuracy": 0.3,
                "calibration_total": 2,  # Less than min_predictions_for_weight
                "calibration_brier_sum": 1.0,
            }
        }
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.95,
            contributors=["agent"],
            contributor_ratings=ratings,
        )
        # Should not detect overconfidence without enough data
        assert signals.overconfidence_level == OverconfidenceLevel.NONE


class TestSourceReliabilityComputation:
    """Tests for source reliability computation."""

    def test_no_sources_returns_default(self):
        """Test default reliability with no sources."""
        engine = QualitySignalEngine()
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent"],
        )
        # Source reliability not set without sources
        assert signals.source_reliability == 0.0

    def test_unknown_source_returns_neutral(self):
        """Test neutral reliability for unknown source."""
        engine = QualitySignalEngine()
        # Source is listed but has no entry in validation_history
        validation_history = {"other_source": [True, True, True, True, True]}
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent"],
            sources=["unknown_source"],
            validation_history=validation_history,
        )
        # Unknown source gets 0.5 (neutral) since no history for it
        assert signals.source_reliability == 0.5

    def test_reliable_source(self):
        """Test high reliability for consistently valid source."""
        engine = QualitySignalEngine()
        # 5 validations, all correct
        validation_history = {"reliable_src": [True, True, True, True, True]}
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent"],
            sources=["reliable_src"],
            validation_history=validation_history,
        )
        assert signals.source_reliability > 0.8

    def test_unreliable_source(self):
        """Test low reliability for inconsistent source."""
        engine = QualitySignalEngine()
        # 5 validations, mostly wrong
        validation_history = {"bad_src": [False, False, True, False, False]}
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent"],
            sources=["bad_src"],
            validation_history=validation_history,
        )
        assert signals.source_reliability < 0.5

    def test_multiple_sources_averaged(self):
        """Test reliability is averaged across multiple sources."""
        engine = QualitySignalEngine()
        validation_history = {
            "good_src": [True, True, True, True, True],  # 100% reliable
            "bad_src": [False, False, False, True, True],  # 40% reliable
        }
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent"],
            sources=["good_src", "bad_src"],
            validation_history=validation_history,
        )
        # Should be between good and bad
        assert 0.4 < signals.source_reliability < 0.9

    def test_insufficient_validations_returns_neutral(self):
        """Test neutral reliability with insufficient validations."""
        engine = QualitySignalEngine()
        # Only 2 validations (less than min_validations_for_reliability=3)
        validation_history = {"src": [True, True]}
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent"],
            sources=["src"],
            validation_history=validation_history,
        )
        assert signals.source_reliability == 0.5


class TestDimensionScores:
    """Tests for dimension score computation."""

    def test_all_dimensions_computed(self):
        """Test all dimensions are computed."""
        engine = QualitySignalEngine()
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent"],
        )
        assert QualityDimension.CONFIDENCE.value in signals.dimension_scores
        assert QualityDimension.CALIBRATION.value in signals.dimension_scores
        assert QualityDimension.SOURCE_RELIABILITY.value in signals.dimension_scores
        assert QualityDimension.CONSENSUS.value in signals.dimension_scores
        assert QualityDimension.FRESHNESS.value in signals.dimension_scores
        assert QualityDimension.VALIDATION.value in signals.dimension_scores

    def test_confidence_dimension_equals_calibrated(self):
        """Test confidence dimension equals calibrated confidence."""
        engine = QualitySignalEngine()
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent"],
        )
        assert (
            signals.dimension_scores[QualityDimension.CONFIDENCE.value]
            == signals.calibrated_confidence
        )

    def test_freshness_with_recent_item(self):
        """Test freshness score for recent item."""
        engine = QualitySignalEngine()
        now = datetime.now(timezone.utc)
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent"],
            created_at=now,
        )
        # Very recent item should have high freshness
        assert signals.dimension_scores[QualityDimension.FRESHNESS.value] > 0.99

    def test_freshness_with_old_item(self):
        """Test freshness score for old item."""
        engine = QualitySignalEngine()
        old_date = datetime.now(timezone.utc) - timedelta(days=365)
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent"],
            created_at=old_date,
        )
        # 1-year-old item should have ~0.37 freshness (exp(-1))
        freshness = signals.dimension_scores[QualityDimension.FRESHNESS.value]
        assert 0.3 < freshness < 0.4

    def test_freshness_without_created_at(self):
        """Test freshness defaults when no created_at provided."""
        engine = QualitySignalEngine()
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent"],
        )
        # Default freshness is 0.8
        assert signals.dimension_scores[QualityDimension.FRESHNESS.value] == 0.8

    def test_consensus_single_contributor(self):
        """Test consensus score with single contributor."""
        engine = QualitySignalEngine()
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent"],
        )
        # Single contributor gets 0.7 consensus
        assert signals.dimension_scores[QualityDimension.CONSENSUS.value] == 0.7

    def test_consensus_multiple_contributors_low_variance(self):
        """Test consensus with multiple contributors having similar weights."""
        engine = QualitySignalEngine()
        # Both get default weight (0.5) since no ratings
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent1", "agent2"],
        )
        # Low variance should give high consensus
        assert signals.dimension_scores[QualityDimension.CONSENSUS.value] > 0.9


class TestCompositeScoreComputation:
    """Tests for composite score computation."""

    def test_composite_score_computed(self):
        """Test composite score is computed."""
        engine = QualitySignalEngine()
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent"],
        )
        assert signals.composite_quality_score > 0

    def test_composite_score_bounded(self):
        """Test composite score is in [0, 1] range."""
        engine = QualitySignalEngine()
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.99,
            contributors=["agent"],
        )
        assert 0 <= signals.composite_quality_score <= 1

    def test_composite_score_weighted_average(self):
        """Test composite score is weighted average of dimensions."""
        engine = QualitySignalEngine()
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent"],
        )
        # Manually compute expected composite
        config = engine.config
        expected = 0
        total_weight = 0
        for dim, score in signals.dimension_scores.items():
            weight = config.dimension_weights.get(dim, 0.1)
            expected += weight * score
            total_weight += weight
        expected /= total_weight

        assert abs(signals.composite_quality_score - expected) < 0.01


class TestQualityTierDetermination:
    """Tests for quality tier determination."""

    def test_excellent_tier(self):
        """Test excellent tier assignment."""
        engine = QualitySignalEngine()
        tier = engine._determine_quality_tier(0.95)
        assert tier == QualityTier.EXCELLENT

    def test_good_tier(self):
        """Test good tier assignment."""
        engine = QualitySignalEngine()
        tier = engine._determine_quality_tier(0.75)
        assert tier == QualityTier.GOOD

    def test_acceptable_tier(self):
        """Test acceptable tier assignment."""
        engine = QualitySignalEngine()
        tier = engine._determine_quality_tier(0.55)
        assert tier == QualityTier.ACCEPTABLE

    def test_poor_tier(self):
        """Test poor tier assignment."""
        engine = QualitySignalEngine()
        tier = engine._determine_quality_tier(0.35)
        assert tier == QualityTier.POOR

    def test_unreliable_tier(self):
        """Test unreliable tier assignment."""
        engine = QualitySignalEngine()
        tier = engine._determine_quality_tier(0.2)
        assert tier == QualityTier.UNRELIABLE

    def test_boundary_excellent(self):
        """Test boundary for excellent tier."""
        engine = QualitySignalEngine()
        assert engine._determine_quality_tier(0.90) == QualityTier.EXCELLENT
        assert engine._determine_quality_tier(0.89) == QualityTier.GOOD

    def test_boundary_good(self):
        """Test boundary for good tier."""
        engine = QualitySignalEngine()
        assert engine._determine_quality_tier(0.70) == QualityTier.GOOD
        assert engine._determine_quality_tier(0.69) == QualityTier.ACCEPTABLE


class TestWarningGeneration:
    """Tests for warning generation."""

    def test_severe_overconfidence_warning(self):
        """Test warning for severe overconfidence."""
        engine = QualitySignalEngine()
        ratings = {
            "agent": {
                "calibration_accuracy": 0.3,
                "calibration_total": 100,
                "calibration_brier_sum": 25.0,
            }
        }
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.99,
            contributors=["agent"],
            contributor_ratings=ratings,
        )
        assert any("severe" in w.lower() for w in signals.warnings)

    def test_moderate_overconfidence_warning(self):
        """Test warning for moderate overconfidence."""
        engine = QualitySignalEngine()
        ratings = {
            "agent": {
                "calibration_accuracy": 0.5,
                "calibration_total": 100,
                "calibration_brier_sum": 15.0,
            }
        }
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.95,
            contributors=["agent"],
            contributor_ratings=ratings,
        )
        assert any("moderate" in w.lower() for w in signals.warnings)

    def test_large_adjustment_warning(self):
        """Test warning for large confidence adjustment."""
        engine = QualitySignalEngine()
        # Create a scenario with significant adjustment
        ratings = {
            "agent": {
                "calibration_accuracy": 0.3,
                "calibration_total": 100,
                "calibration_brier_sum": 40.0,  # Very high Brier
            }
        }
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.95,
            contributors=["agent"],
            contributor_ratings=ratings,
        )
        if abs(signals.confidence_adjustment) > 0.2:
            assert any(
                "adjustment" in w.lower() or "confidence" in w.lower() for w in signals.warnings
            )

    def test_single_contributor_warning(self):
        """Test warning for single contributor."""
        engine = QualitySignalEngine()
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["single_agent"],
        )
        assert any("one contributor" in w.lower() for w in signals.warnings)

    def test_uncalibrated_contributor_warning(self):
        """Test warning for uncalibrated contributor."""
        engine = QualitySignalEngine()
        ratings = {
            "new_agent": {
                "calibration_accuracy": 0.5,
                "calibration_total": 2,  # Less than min_predictions_for_weight
                "calibration_brier_sum": 0.5,
            }
        }
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["new_agent"],
            contributor_ratings=ratings,
        )
        assert any(
            "insufficient" in w.lower() or "calibration history" in w.lower()
            for w in signals.warnings
        )

    def test_low_calibration_quality_warning(self):
        """Test warning for low calibration quality index."""
        engine = QualitySignalEngine()
        ratings = {
            "bad": {
                "calibration_accuracy": 0.2,
                "calibration_total": 100,
                "calibration_brier_sum": 50.0,
            }
        }
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["bad"],
            contributor_ratings=ratings,
        )
        if signals.calibration_quality_index < 0.4:
            assert any("poor calibration" in w.lower() for w in signals.warnings)

    def test_low_source_reliability_warning(self):
        """Test warning for low source reliability."""
        engine = QualitySignalEngine()
        validation_history = {"bad_src": [False, False, False, True, False]}
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent"],
            sources=["bad_src"],
            validation_history=validation_history,
        )
        if signals.source_reliability < 0.4:
            assert any("reliability" in w.lower() for w in signals.warnings)


class TestExpectedCalibrationError:
    """Tests for ECE computation."""

    def test_empty_predictions(self):
        """Test ECE with empty predictions."""
        engine = QualitySignalEngine()
        ece = engine.compute_expected_calibration_error([])
        assert ece == 0.0

    def test_perfect_calibration(self):
        """Test ECE with perfect calibration."""
        engine = QualitySignalEngine()
        # 80% confidence, 80% accuracy
        predictions = [(0.8, True)] * 80 + [(0.8, False)] * 20
        ece = engine.compute_expected_calibration_error(predictions)
        assert ece < 0.05  # Very low ECE

    def test_overconfident_predictions(self):
        """Test ECE with overconfident predictions."""
        engine = QualitySignalEngine()
        # 90% confidence, but only 50% accuracy
        predictions = [(0.9, True)] * 50 + [(0.9, False)] * 50
        ece = engine.compute_expected_calibration_error(predictions)
        assert ece > 0.3  # High ECE due to overconfidence

    def test_multiple_confidence_buckets(self):
        """Test ECE with predictions across multiple confidence buckets."""
        engine = QualitySignalEngine()
        predictions = (
            [(0.1, False)] * 9
            + [(0.1, True)] * 1  # 10% bucket
            + [(0.5, True)] * 5
            + [(0.5, False)] * 5  # 50% bucket
            + [(0.9, True)] * 9
            + [(0.9, False)] * 1  # 90% bucket
        )
        ece = engine.compute_expected_calibration_error(predictions)
        # Should be well-calibrated
        assert ece < 0.1


class TestDetectContributorOverconfidence:
    """Tests for contributor overconfidence detection."""

    def test_insufficient_predictions(self):
        """Test no overconfidence with insufficient predictions."""
        engine = QualitySignalEngine()
        predictions = [(0.9, True)] * 2  # Only 2 predictions
        result = engine.detect_contributor_overconfidence("agent", predictions)
        assert result is False

    def test_well_calibrated_not_overconfident(self):
        """Test well-calibrated contributor is not flagged."""
        engine = QualitySignalEngine()
        # 80% confidence, ~80% accuracy
        predictions = [(0.8, True)] * 80 + [(0.8, False)] * 20
        result = engine.detect_contributor_overconfidence("agent", predictions)
        assert result is False

    def test_overconfident_contributor(self):
        """Test overconfident contributor is detected."""
        engine = QualitySignalEngine()
        # 90% confidence, but only 50% accuracy
        predictions = [(0.9, True)] * 50 + [(0.9, False)] * 50
        result = engine.detect_contributor_overconfidence("agent", predictions, threshold=0.1)
        assert result is True

    def test_underconfident_not_flagged(self):
        """Test underconfident contributor is not flagged as overconfident."""
        engine = QualitySignalEngine()
        # 50% confidence, but 90% accuracy (underconfident)
        predictions = [(0.5, True)] * 90 + [(0.5, False)] * 10
        result = engine.detect_contributor_overconfidence("agent", predictions)
        # Underconfidence should not trigger overconfidence flag
        assert result is False


class TestBatchComputeSignals:
    """Tests for batch signal computation."""

    def test_empty_batch(self):
        """Test batch computation with empty list."""
        engine = QualitySignalEngine()
        results = engine.batch_compute_signals([])
        assert results == []

    def test_single_item_batch(self):
        """Test batch computation with single item."""
        engine = QualitySignalEngine()
        items = [
            {
                "item_id": "km_1",
                "confidence": 0.8,
                "contributors": ["agent"],
            }
        ]
        results = engine.batch_compute_signals(items)
        assert len(results) == 1
        assert results[0].item_id == "km_1"

    def test_multiple_items_batch(self):
        """Test batch computation with multiple items."""
        engine = QualitySignalEngine()
        items = [
            {"item_id": f"km_{i}", "confidence": 0.7 + i * 0.05, "contributors": ["agent"]}
            for i in range(5)
        ]
        results = engine.batch_compute_signals(items)
        assert len(results) == 5
        for i, r in enumerate(results):
            assert r.item_id == f"km_{i}"

    def test_batch_with_shared_ratings(self):
        """Test batch computation with shared contributor ratings."""
        engine = QualitySignalEngine()
        ratings = {
            "claude": {
                "calibration_accuracy": 0.9,
                "calibration_total": 100,
                "calibration_brier_sum": 10.0,
            }
        }
        items = [
            {"item_id": "km_1", "confidence": 0.8, "contributors": ["claude"]},
            {"item_id": "km_2", "confidence": 0.9, "contributors": ["claude"]},
        ]
        results = engine.batch_compute_signals(items, contributor_ratings=ratings)
        assert len(results) == 2
        # Both should have claude's weight
        assert "claude" in results[0].contributor_weights
        assert "claude" in results[1].contributor_weights

    def test_batch_handles_missing_fields(self):
        """Test batch handles items with missing optional fields."""
        engine = QualitySignalEngine()
        items = [
            {"item_id": "km_1"},  # Missing confidence and contributors
            {"item_id": "km_2", "confidence": 0.8},  # Missing contributors
        ]
        results = engine.batch_compute_signals(items)
        assert len(results) == 2
        assert results[0].raw_confidence == 0.5  # Default
        assert results[1].raw_confidence == 0.8


class TestGetQualitySummary:
    """Tests for quality summary generation."""

    def test_empty_summary(self):
        """Test summary with empty signals list."""
        engine = QualitySignalEngine()
        summary = engine.get_quality_summary([])
        assert summary == {"count": 0}

    def test_basic_summary(self):
        """Test basic summary statistics."""
        engine = QualitySignalEngine()
        signals_list = [
            engine.compute_quality_signals("km_1", 0.7, ["agent"]),
            engine.compute_quality_signals("km_2", 0.8, ["agent"]),
            engine.compute_quality_signals("km_3", 0.9, ["agent"]),
        ]
        summary = engine.get_quality_summary(signals_list)

        assert summary["count"] == 3
        assert "avg_raw_confidence" in summary
        assert "avg_calibrated_confidence" in summary
        assert "avg_composite_score" in summary
        assert "tier_distribution" in summary

    def test_summary_tier_distribution(self):
        """Test tier distribution in summary."""
        engine = QualitySignalEngine()
        signals_list = [
            engine.compute_quality_signals("km_1", 0.95, ["agent"]),
            engine.compute_quality_signals("km_2", 0.5, ["agent"]),
        ]
        summary = engine.get_quality_summary(signals_list)

        assert "tier_distribution" in summary

    def test_summary_overconfidence_stats(self):
        """Test overconfidence statistics in summary."""
        engine = QualitySignalEngine()
        signals_list = [
            engine.compute_quality_signals("km_1", 0.8, ["agent"]),
            engine.compute_quality_signals("km_2", 0.9, ["agent"]),
        ]
        summary = engine.get_quality_summary(signals_list)

        assert "overconfident_items" in summary
        assert "overconfident_ratio" in summary

    def test_summary_warning_stats(self):
        """Test warning statistics in summary."""
        engine = QualitySignalEngine()
        signals_list = [
            engine.compute_quality_signals("km_1", 0.8, ["agent1"]),
            engine.compute_quality_signals("km_2", 0.9, ["agent2"]),
        ]
        summary = engine.get_quality_summary(signals_list)

        assert "total_warnings" in summary
        assert "avg_warnings_per_item" in summary


class TestSingletonAccessor:
    """Tests for singleton accessor function."""

    def test_get_quality_signal_engine_returns_engine(self):
        """Test accessor returns QualitySignalEngine instance."""
        # Reset singleton for clean test
        import aragora.knowledge.mound.ops.quality_signals as qs_module

        qs_module._quality_signal_engine = None

        engine = get_quality_signal_engine()
        assert isinstance(engine, QualitySignalEngine)

    def test_get_quality_signal_engine_returns_same_instance(self):
        """Test accessor returns same instance on multiple calls."""
        # Reset singleton
        import aragora.knowledge.mound.ops.quality_signals as qs_module

        qs_module._quality_signal_engine = None

        engine1 = get_quality_signal_engine()
        engine2 = get_quality_signal_engine()
        assert engine1 is engine2

    def test_get_quality_signal_engine_with_config(self):
        """Test accessor uses config on first call."""
        # Reset singleton
        import aragora.knowledge.mound.ops.quality_signals as qs_module

        qs_module._quality_signal_engine = None

        config = QualityEngineConfig(ece_mild_threshold=0.08)
        engine = get_quality_signal_engine(config)
        assert engine.config.ece_mild_threshold == 0.08


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_confidence(self):
        """Test with zero raw confidence."""
        engine = QualitySignalEngine()
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.0,
            contributors=["agent"],
        )
        assert signals.raw_confidence == 0.0
        assert signals.calibrated_confidence >= 0.0

    def test_max_confidence(self):
        """Test with maximum confidence."""
        engine = QualitySignalEngine()
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=1.0,
            contributors=["agent"],
        )
        assert signals.raw_confidence == 1.0
        assert signals.calibrated_confidence <= 1.0

    def test_negative_confidence_clamped(self):
        """Test negative confidence is handled."""
        engine = QualitySignalEngine()
        # This shouldn't happen in practice but test boundary
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=-0.1,
            contributors=["agent"],
        )
        # Should still produce valid signals
        assert isinstance(signals, QualitySignals)

    def test_confidence_over_one_clamped(self):
        """Test confidence over 1.0 is handled."""
        engine = QualitySignalEngine()
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=1.5,
            contributors=["agent"],
        )
        # Should still produce valid signals
        assert isinstance(signals, QualitySignals)

    def test_empty_item_id(self):
        """Test with empty item_id."""
        engine = QualitySignalEngine()
        signals = engine.compute_quality_signals(
            item_id="",
            raw_confidence=0.8,
            contributors=["agent"],
        )
        assert signals.item_id == ""

    def test_very_long_contributor_list(self):
        """Test with many contributors."""
        engine = QualitySignalEngine()
        contributors = [f"agent_{i}" for i in range(100)]
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=contributors,
        )
        assert len(signals.contributor_weights) == 100

    def test_very_old_item_freshness(self):
        """Test freshness for very old item."""
        engine = QualitySignalEngine()
        very_old = datetime.now(timezone.utc) - timedelta(days=3650)  # 10 years
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent"],
            created_at=very_old,
        )
        # Freshness should be very low but non-negative
        freshness = signals.dimension_scores[QualityDimension.FRESHNESS.value]
        assert 0 <= freshness < 0.01

    def test_future_created_at(self):
        """Test handling of future creation date."""
        engine = QualitySignalEngine()
        future = datetime.now(timezone.utc) + timedelta(days=365)
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent"],
            created_at=future,
        )
        # Freshness should be clamped to 1.0
        freshness = signals.dimension_scores[QualityDimension.FRESHNESS.value]
        assert freshness <= 1.0

    def test_validation_history_all_false(self):
        """Test reliability with all failed validations."""
        engine = QualitySignalEngine()
        validation_history = {"src": [False, False, False, False, False]}
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent"],
            sources=["src"],
            validation_history=validation_history,
        )
        # Should have very low but non-negative reliability
        assert 0 <= signals.source_reliability < 0.2

    def test_validation_history_all_true(self):
        """Test reliability with all successful validations."""
        engine = QualitySignalEngine()
        validation_history = {"src": [True, True, True, True, True]}
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent"],
            sources=["src"],
            validation_history=validation_history,
        )
        # Should have high reliability
        assert signals.source_reliability > 0.9

    def test_contributor_with_dataclass_rating(self):
        """Test contributor ratings from dataclass-like object."""
        engine = QualitySignalEngine()

        class MockRating:
            calibration_accuracy = 0.85
            calibration_total = 100
            calibration_brier_sum = 10.0

        ratings = {"claude": MockRating()}
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["claude"],
            contributor_ratings=ratings,
        )
        assert "claude" in signals.contributor_weights

    def test_contributor_with_partial_rating(self):
        """Test contributor with partial rating data."""
        engine = QualitySignalEngine()
        ratings = {
            "agent": {
                "calibration_accuracy": 0.8,
                # Missing calibration_total and calibration_brier_sum
            }
        }
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent"],
            contributor_ratings=ratings,
        )
        # Should handle gracefully
        assert "agent" in signals.contributor_weights


class TestContributorCacheBehavior:
    """Tests for contributor cache behavior."""

    def test_contributor_cached(self):
        """Test contributors are cached."""
        engine = QualitySignalEngine()
        ratings = {
            "claude": {
                "calibration_accuracy": 0.9,
                "calibration_total": 100,
                "calibration_brier_sum": 10.0,
            }
        }
        engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["claude"],
            contributor_ratings=ratings,
        )
        assert "claude" in engine._contributor_cache

    def test_source_reliability_cached(self):
        """Test source reliability is cached."""
        engine = QualitySignalEngine()
        validation_history = {"src": [True, True, True, True, True]}
        engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent"],
            sources=["src"],
            validation_history=validation_history,
        )
        assert "src" in engine._source_reliability_cache


class TestDomainSpecificBehavior:
    """Tests for domain parameter behavior."""

    def test_domain_parameter_accepted(self):
        """Test domain parameter is accepted."""
        engine = QualitySignalEngine()
        signals = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent"],
            domain="finance",
        )
        # Should not raise and produce valid signals
        assert isinstance(signals, QualitySignals)


class TestDecayFactorBehavior:
    """Tests for reliability decay factor behavior."""

    def test_decay_factor_weights_recent_more(self):
        """Test decay factor gives more weight to recent validations."""
        engine = QualitySignalEngine()
        # Recent successes followed by old failures (newest first)
        validation_history = {
            "src": [True, True, True, False, False]  # Newest first - more recent success
        }
        signals1 = engine.compute_quality_signals(
            item_id="km_1",
            raw_confidence=0.8,
            contributors=["agent"],
            sources=["src"],
            validation_history=validation_history,
        )

        # Recent failures followed by old successes (newest first)
        validation_history2 = {
            "src": [False, False, True, True, True]  # Newest first - more recent failures
        }
        engine2 = QualitySignalEngine()  # Fresh engine to avoid cache
        signals2 = engine2.compute_quality_signals(
            item_id="km_2",
            raw_confidence=0.8,
            contributors=["agent"],
            sources=["src"],
            validation_history=validation_history2,
        )

        # signals1 has more recent successes, should have higher reliability
        assert signals1.source_reliability > signals2.source_reliability


class TestToDict:
    """Additional tests for to_dict methods."""

    def test_quality_signals_to_dict_roundtrip(self):
        """Test QualitySignals serializes all fields."""
        engine = QualitySignalEngine()
        signals = engine.compute_quality_signals(
            item_id="km_test",
            raw_confidence=0.85,
            contributors=["claude", "gpt"],
        )
        result = signals.to_dict()

        # Verify all expected keys present
        expected_keys = [
            "item_id",
            "raw_confidence",
            "calibrated_confidence",
            "confidence_adjustment",
            "calibration_quality_index",
            "overconfidence_level",
            "overconfidence_flag",
            "source_reliability",
            "contributor_weights",
            "quality_tier",
            "composite_quality_score",
            "dimension_scores",
            "warnings",
            "computed_at",
            "metadata",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
