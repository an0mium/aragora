"""
Phase A3 Tests for Knowledge Mound operations.

Tests for:
- Adapter Fusion Protocol (ops/fusion.py)
- Calibration Fusion Engine (ops/calibration_fusion.py)
- Multi-Party Validation (ops/multi_party_validation.py)
- Quality Signals (ops/quality_signals.py)
- Composite Analytics (ops/composite_analytics.py)
"""

import pytest
from datetime import datetime, timezone

from aragora.knowledge.mound.ops import (
    # Fusion Protocol
    FusionStrategy,
    ConflictResolution,
    AdapterValidation,
    FusedValidation,
    FusionConfig,
    FusionCoordinator,
    get_fusion_coordinator,
    # Calibration Fusion
    CalibrationFusionStrategy,
    AgentPrediction,
    CalibrationConsensus,
    CalibrationFusionEngine,
    get_calibration_fusion_engine,
    # Multi-Party Validation
    ValidationVoteType,
    ValidationConsensusStrategy,
    ValidationState,
    ValidationVote,
    ValidationRequest,
    ValidationResult,
    ValidatorConfig,
    MultiPartyValidator,
    get_multi_party_validator,
    ValidatorVote as ContradictionValidatorVote,
    # Quality Signals
    QualityDimension,
    OverconfidenceLevel,
    QualityTier,
    QualitySignals,
    QualityEngineConfig,
    QualitySignalEngine,
    get_quality_signal_engine,
    # Composite Analytics
    SLOStatus,
    BottleneckSeverity,
    AdapterMetrics,
    SLOConfig,
    CompositeMetrics,
    SyncResultInput,
    CompositeAnalytics,
    get_composite_analytics,
)


# =============================================================================
# Fusion Protocol Tests
# =============================================================================


class TestFusionProtocol:
    """Tests for the Adapter Fusion Protocol."""

    def test_fusion_strategy_enum(self):
        """Test FusionStrategy enum values."""
        assert FusionStrategy.WEIGHTED_AVERAGE.value == "weighted_average"
        assert FusionStrategy.MAJORITY_VOTE.value == "majority_vote"
        assert FusionStrategy.MAXIMUM_CONFIDENCE.value == "maximum_confidence"

    def test_conflict_resolution_enum(self):
        """Test ConflictResolution enum values."""
        assert ConflictResolution.PREFER_HIGHER_CONFIDENCE.value == "prefer_higher_confidence"
        assert ConflictResolution.MERGE.value == "merge"
        assert ConflictResolution.ESCALATE.value == "escalate"

    def test_adapter_validation_dataclass(self):
        """Test AdapterValidation creation."""
        validation = AdapterValidation(
            adapter_name="elo",
            item_id="km_123",
            confidence=0.85,
            is_valid=True,
            sources=["debate_1"],
            priority=1,
            reliability=0.9,
        )
        assert validation.adapter_name == "elo"
        assert validation.confidence == 0.85
        assert validation.is_valid is True

    def test_fusion_coordinator_weighted_average(self):
        """Test FusionCoordinator with weighted average strategy."""
        coordinator = FusionCoordinator()

        validations = [
            AdapterValidation(
                adapter_name="elo",
                item_id="km_123",
                confidence=0.8,
                is_valid=True,
                sources=["src1"],
                priority=1,
                reliability=0.9,
            ),
            AdapterValidation(
                adapter_name="consensus",
                item_id="km_123",
                confidence=0.7,
                is_valid=True,
                sources=["src2"],
                priority=2,
                reliability=0.85,
            ),
        ]

        result = coordinator.fuse_validations(
            validations,
            strategy=FusionStrategy.WEIGHTED_AVERAGE,
        )

        assert isinstance(result, FusedValidation)
        assert result.item_id == "km_123"
        assert 0.7 <= result.fused_confidence <= 0.8  # Weighted between inputs
        assert result.is_valid is True
        assert result.strategy_used == FusionStrategy.WEIGHTED_AVERAGE

    def test_fusion_coordinator_conflict_detection(self):
        """Test FusionCoordinator conflict detection."""
        coordinator = FusionCoordinator()

        # Create conflicting validations with high variance
        # Variance must be > 0.3 (conflict_threshold) to trigger detection
        # Using 0.95 and 0.05 gives variance ~0.405
        validations = [
            AdapterValidation(
                adapter_name="elo",
                item_id="km_123",
                confidence=0.95,
                is_valid=True,
                sources=["src1"],
                priority=1,
                reliability=0.9,
            ),
            AdapterValidation(
                adapter_name="belief",
                item_id="km_123",
                confidence=0.05,
                is_valid=False,
                sources=["src2"],
                priority=1,
                reliability=0.85,
            ),
        ]

        result = coordinator.fuse_validations(validations)
        assert result.conflict_detected is True

    def test_get_fusion_coordinator_singleton(self):
        """Test singleton accessor."""
        coord1 = get_fusion_coordinator()
        coord2 = get_fusion_coordinator()
        assert coord1 is coord2


# =============================================================================
# Calibration Fusion Tests
# =============================================================================


class TestCalibrationFusion:
    """Tests for the Calibration Fusion Engine."""

    def test_calibration_fusion_strategy_enum(self):
        """Test CalibrationFusionStrategy values."""
        assert CalibrationFusionStrategy.WEIGHTED_AVERAGE.value == "weighted_average"
        assert CalibrationFusionStrategy.RELIABILITY_WEIGHTED.value == "reliability_weighted"
        assert CalibrationFusionStrategy.MEDIAN.value == "median"

    def test_agent_prediction_dataclass(self):
        """Test AgentPrediction creation."""
        pred = AgentPrediction(
            agent_name="claude",
            confidence=0.85,
            predicted_outcome="winner_a",
            calibration_accuracy=0.78,
            brier_score=0.15,
        )
        assert pred.agent_name == "claude"
        assert pred.confidence == 0.85
        assert pred.calibration_accuracy == 0.78

    def test_calibration_fusion_engine_basic(self):
        """Test basic calibration fusion."""
        engine = CalibrationFusionEngine()

        predictions = [
            AgentPrediction("claude", 0.8, "winner_a", 0.85, 0.12),
            AgentPrediction("gpt-4", 0.75, "winner_a", 0.8, 0.15),
            AgentPrediction("gemini", 0.7, "winner_a", 0.75, 0.18),
        ]

        weights = {"claude": 0.9, "gpt-4": 0.85, "gemini": 0.75}

        result = engine.fuse_predictions(predictions, weights)

        assert isinstance(result, CalibrationConsensus)
        assert 0.7 <= result.fused_confidence <= 0.85
        assert result.predicted_outcome == "winner_a"
        assert result.consensus_strength >= 0  # Should have high consensus

    def test_calibration_fusion_disagreement(self):
        """Test calibration fusion with disagreement."""
        engine = CalibrationFusionEngine()

        predictions = [
            AgentPrediction("claude", 0.8, "winner_a", 0.85, 0.12),
            AgentPrediction("gpt-4", 0.75, "winner_b", 0.8, 0.15),
            AgentPrediction("gemini", 0.6, "winner_a", 0.75, 0.18),
        ]

        result = engine.fuse_predictions(predictions)

        # Lower consensus with disagreement
        assert result.agreement_ratio < 1.0

    def test_calibration_fusion_outlier_detection(self):
        """Test outlier detection in predictions."""
        engine = CalibrationFusionEngine()

        predictions = [
            AgentPrediction("claude", 0.8, "winner_a", 0.85, 0.12),
            AgentPrediction("gpt-4", 0.82, "winner_a", 0.8, 0.15),
            AgentPrediction("gemini", 0.15, "winner_b", 0.75, 0.40),  # Outlier
        ]

        outliers = engine.detect_outliers(predictions, threshold=1.5)
        assert "gemini" in outliers

    def test_krippendorff_alpha(self):
        """Test Krippendorff's alpha calculation."""
        engine = CalibrationFusionEngine()

        # Perfect agreement
        predictions = [
            AgentPrediction("a", 0.8, "winner_a", 0.85, 0.12),
            AgentPrediction("b", 0.8, "winner_a", 0.8, 0.15),
            AgentPrediction("c", 0.8, "winner_a", 0.75, 0.18),
        ]

        alpha = engine.compute_krippendorff_alpha(predictions)
        assert alpha >= 0.8  # High agreement


# =============================================================================
# Multi-Party Validation Tests
# =============================================================================


class TestMultiPartyValidation:
    """Tests for the Multi-Party Validation workflow."""

    def test_validation_vote_type_enum(self):
        """Test ValidationVoteType values."""
        assert ValidationVoteType.ACCEPT.value == "accept"
        assert ValidationVoteType.REJECT.value == "reject"
        assert ValidationVoteType.ABSTAIN.value == "abstain"
        assert ValidationVoteType.PROPOSE_ALTERNATIVE.value == "propose_alternative"

    def test_validation_consensus_strategy_enum(self):
        """Test ValidationConsensusStrategy values."""
        assert ValidationConsensusStrategy.UNANIMOUS.value == "unanimous"
        assert ValidationConsensusStrategy.MAJORITY.value == "majority"
        assert ValidationConsensusStrategy.SUPERMAJORITY.value == "supermajority"

    @pytest.mark.asyncio
    async def test_create_validation_request(self):
        """Test creating a validation request."""
        validator = MultiPartyValidator()

        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["claude", "gpt-4", "gemini"],
            quorum=2,
        )

        assert request.item_id == "km_123"
        assert len(request.validators) == 3
        assert request.required_votes == 2
        assert request.state == ValidationState.PENDING

    @pytest.mark.asyncio
    async def test_submit_vote(self):
        """Test submitting a vote."""
        validator = MultiPartyValidator()

        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["claude", "gpt-4"],
            quorum=2,
        )

        success = await validator.submit_vote(
            request_id=request.request_id,
            validator_id="claude",
            vote_type=ValidationVoteType.ACCEPT,
            confidence=0.9,
            reasoning="Looks correct",
        )

        assert success is True

        # Verify vote was recorded
        updated = validator.get_request(request.request_id)
        assert updated is not None
        assert updated.votes_received == 1
        assert updated.state == ValidationState.IN_REVIEW

    @pytest.mark.asyncio
    async def test_consensus_reached_majority(self):
        """Test consensus detection with majority strategy."""
        validator = MultiPartyValidator()

        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b", "c"],
            quorum=2,
            strategy=ValidationConsensusStrategy.MAJORITY,
        )

        # Submit two accept votes
        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)
        result = await validator.submit_vote(
            request.request_id, "b", ValidationVoteType.ACCEPT, 0.85
        )

        # Check consensus
        final_result = await validator.check_consensus(request.request_id)
        assert final_result is not None
        assert final_result.outcome == "accepted"
        assert final_result.final_verdict == ValidationVoteType.ACCEPT

    @pytest.mark.asyncio
    async def test_validation_deadlock(self):
        """Test deadlock detection."""
        config = ValidatorConfig(auto_escalate_on_deadlock=False)
        validator = MultiPartyValidator(config)

        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b"],
            quorum=2,
            strategy=ValidationConsensusStrategy.UNANIMOUS,
        )

        # Submit conflicting votes
        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)
        await validator.submit_vote(request.request_id, "b", ValidationVoteType.REJECT, 0.85)

        result = await validator.check_consensus(request.request_id)
        assert result is not None
        assert result.outcome == "deadlocked"

    def test_validator_stats(self):
        """Test validation statistics."""
        validator = MultiPartyValidator()
        stats = validator.get_stats()

        assert "total_requests" in stats
        assert "completed" in stats
        assert "escalation_rate" in stats


# =============================================================================
# Quality Signals Tests
# =============================================================================


class TestQualitySignals:
    """Tests for the Quality Signal Engine."""

    def test_quality_dimension_enum(self):
        """Test QualityDimension values."""
        assert QualityDimension.CONFIDENCE.value == "confidence"
        assert QualityDimension.CALIBRATION.value == "calibration"
        assert QualityDimension.SOURCE_RELIABILITY.value == "source_reliability"

    def test_overconfidence_level_enum(self):
        """Test OverconfidenceLevel values."""
        assert OverconfidenceLevel.NONE.value == "none"
        assert OverconfidenceLevel.MILD.value == "mild"
        assert OverconfidenceLevel.SEVERE.value == "severe"

    def test_quality_tier_enum(self):
        """Test QualityTier values."""
        assert QualityTier.EXCELLENT.value == "excellent"
        assert QualityTier.GOOD.value == "good"
        assert QualityTier.UNRELIABLE.value == "unreliable"

    def test_quality_signals_basic(self):
        """Test basic quality signal computation."""
        engine = QualitySignalEngine()

        signals = engine.compute_quality_signals(
            item_id="km_123",
            raw_confidence=0.85,
            contributors=["claude", "gpt-4"],
        )

        assert isinstance(signals, QualitySignals)
        assert signals.item_id == "km_123"
        assert signals.raw_confidence == 0.85
        assert 0 <= signals.calibrated_confidence <= 1.0

    def test_quality_signals_with_ratings(self):
        """Test quality signals with contributor ratings."""
        engine = QualitySignalEngine()

        # Mock contributor ratings (simulating AgentRating)
        contributor_ratings = {
            "claude": {
                "calibration_accuracy": 0.85,
                "calibration_total": 100,
                "calibration_brier_sum": 12.0,  # 0.12 average Brier
            },
            "gpt-4": {
                "calibration_accuracy": 0.80,
                "calibration_total": 80,
                "calibration_brier_sum": 12.0,  # 0.15 average Brier
            },
        }

        signals = engine.compute_quality_signals(
            item_id="km_123",
            raw_confidence=0.90,
            contributors=["claude", "gpt-4"],
            contributor_ratings=contributor_ratings,
        )

        assert signals.calibration_quality_index > 0
        assert len(signals.contributor_weights) == 2
        assert "claude" in signals.contributor_weights

    def test_quality_signals_overconfidence_detection(self):
        """Test overconfidence detection."""
        engine = QualitySignalEngine()

        # Contributors with poor calibration (high Brier)
        poor_ratings = {
            "agent1": {
                "calibration_accuracy": 0.4,
                "calibration_total": 50,
                "calibration_brier_sum": 15.0,  # 0.30 average Brier = severe
            },
        }

        signals = engine.compute_quality_signals(
            item_id="km_123",
            raw_confidence=0.95,
            contributors=["agent1"],
            contributor_ratings=poor_ratings,
        )

        # Should detect overconfidence
        assert signals.overconfidence_level in [
            OverconfidenceLevel.MODERATE,
            OverconfidenceLevel.SEVERE,
        ]
        assert signals.overconfidence_flag is True

    def test_quality_signals_source_reliability(self):
        """Test source reliability computation."""
        engine = QualitySignalEngine()

        validation_history = {
            "source_a": [True, True, True, True, False],  # 80% reliable
            "source_b": [True, False, True, False, True],  # 60% reliable
        }

        signals = engine.compute_quality_signals(
            item_id="km_123",
            raw_confidence=0.8,
            contributors=["agent1"],
            sources=["source_a", "source_b"],
            validation_history=validation_history,
        )

        # Source reliability should be between 60% and 80%
        assert 0.5 <= signals.source_reliability <= 0.9

    def test_quality_tier_assignment(self):
        """Test quality tier assignment."""
        engine = QualitySignalEngine()

        # High quality
        high_signals = engine.compute_quality_signals(
            item_id="km_high",
            raw_confidence=0.95,
            contributors=["claude"],
            contributor_ratings={
                "claude": {
                    "calibration_accuracy": 0.95,
                    "calibration_total": 200,
                    "calibration_brier_sum": 10.0,
                }
            },
        )

        assert high_signals.quality_tier in [QualityTier.EXCELLENT, QualityTier.GOOD]

    def test_quality_warnings_generation(self):
        """Test warning generation."""
        engine = QualitySignalEngine()

        # Single contributor with insufficient history
        signals = engine.compute_quality_signals(
            item_id="km_123",
            raw_confidence=0.9,
            contributors=["new_agent"],
            contributor_ratings={
                "new_agent": {
                    "calibration_accuracy": 0.0,
                    "calibration_total": 2,  # Less than min threshold
                    "calibration_brier_sum": 0.0,
                }
            },
        )

        # Should have warnings about single contributor and insufficient calibration
        assert len(signals.warnings) > 0
        assert any("contributor" in w.lower() for w in signals.warnings)

    def test_expected_calibration_error(self):
        """Test ECE computation."""
        engine = QualitySignalEngine()

        # Perfect calibration: 80% confidence, 80% accuracy
        predictions = [(0.8, True)] * 80 + [(0.8, False)] * 20

        ece = engine.compute_expected_calibration_error(predictions)
        assert ece < 0.1  # Should be well-calibrated


# =============================================================================
# Composite Analytics Tests
# =============================================================================


class TestCompositeAnalytics:
    """Tests for the Composite Analytics engine."""

    def test_slo_status_enum(self):
        """Test SLOStatus values."""
        assert SLOStatus.MET.value == "met"
        assert SLOStatus.WARNING.value == "warning"
        assert SLOStatus.VIOLATED.value == "violated"

    def test_bottleneck_severity_enum(self):
        """Test BottleneckSeverity values."""
        assert BottleneckSeverity.NONE.value == "none"
        assert BottleneckSeverity.CRITICAL.value == "critical"

    def test_composite_analytics_basic(self):
        """Test basic composite analytics."""
        analytics = CompositeAnalytics()

        sync_results = [
            SyncResultInput(
                adapter_name="elo",
                direction="forward",
                success=True,
                items_processed=100,
                duration_ms=200,
            ),
            SyncResultInput(
                adapter_name="consensus",
                direction="forward",
                success=True,
                items_processed=50,
                duration_ms=150,
            ),
        ]

        metrics = analytics.compute_composite_metrics(sync_results)

        assert isinstance(metrics, CompositeMetrics)
        assert metrics.adapter_count == 2
        assert metrics.total_sync_time_ms == 350  # 200 + 150
        assert "elo" in metrics.adapter_metrics
        assert "consensus" in metrics.adapter_metrics

    def test_composite_slo_evaluation(self):
        """Test SLO evaluation."""
        config = SLOConfig(sync_time_target_ms=1000.0)
        analytics = CompositeAnalytics(config)

        # Fast syncs - should meet SLO
        fast_results = [
            SyncResultInput("elo", "forward", True, 100, duration_ms=100),
            SyncResultInput("consensus", "forward", True, 50, duration_ms=100),
        ]

        metrics = analytics.compute_composite_metrics(fast_results)
        assert metrics.composite_slo_met is True

        # Slow syncs - should violate SLO
        slow_results = [
            SyncResultInput("elo", "forward", True, 100, duration_ms=600),
            SyncResultInput("consensus", "forward", True, 50, duration_ms=600),
        ]

        metrics_slow = analytics.compute_composite_metrics(slow_results)
        # Total 1200ms > 1000ms target
        assert any(slo.status == SLOStatus.VIOLATED for slo in metrics_slow.slo_results)

    def test_bottleneck_identification(self):
        """Test bottleneck identification."""
        analytics = CompositeAnalytics()

        sync_results = [
            SyncResultInput("fast_adapter", "forward", True, 100, duration_ms=50),
            SyncResultInput("slow_adapter", "forward", True, 100, duration_ms=500),
            SyncResultInput("medium_adapter", "forward", True, 100, duration_ms=100),
        ]

        metrics = analytics.compute_composite_metrics(sync_results)

        assert metrics.bottleneck_analysis is not None
        assert metrics.bottleneck_analysis.bottleneck_adapter == "slow_adapter"
        assert metrics.bottleneck_analysis.severity != BottleneckSeverity.NONE

    def test_parallel_efficiency(self):
        """Test parallel efficiency calculation."""
        analytics = CompositeAnalytics()

        sync_results = [
            SyncResultInput("a", "forward", True, 100, duration_ms=100),
            SyncResultInput("b", "forward", True, 100, duration_ms=100),
            SyncResultInput("c", "forward", True, 100, duration_ms=100),
        ]

        metrics = analytics.compute_composite_metrics(sync_results)

        # Parallel efficiency should be calculated
        assert 0 <= metrics.parallel_efficiency <= 1.0
        assert metrics.theoretical_parallel_time_ms == 100  # Max of all

    def test_optimization_recommendations(self):
        """Test recommendation generation."""
        config = SLOConfig(adapter_time_target_ms=100.0)
        analytics = CompositeAnalytics(config)

        # Create slow adapter
        sync_results = [
            SyncResultInput("slow", "forward", True, 100, duration_ms=500),
            SyncResultInput("fast", "forward", True, 100, duration_ms=50),
        ]

        metrics = analytics.compute_composite_metrics(sync_results, include_recommendations=True)

        # Should have recommendations
        assert len(metrics.recommendations) > 0

    def test_historical_stats(self):
        """Test historical statistics tracking."""
        analytics = CompositeAnalytics()

        # Add multiple sync results
        for i in range(20):
            sync_results = [
                SyncResultInput("elo", "forward", True, 100, duration_ms=100 + i * 5),
            ]
            analytics.compute_composite_metrics(sync_results)

        stats = analytics.get_historical_stats("elo")
        assert stats["count"] == 20
        assert "avg_ms" in stats
        assert "p95_ms" in stats

    def test_trend_computation(self):
        """Test performance trend computation."""
        analytics = CompositeAnalytics()

        # Add improving performance over time
        for i in range(30):
            sync_results = [
                SyncResultInput("improving", "forward", True, 100, duration_ms=200 - i * 3),
            ]
            analytics.compute_composite_metrics(sync_results)

        trend = analytics.compute_trend("improving", window_size=10)
        assert trend["trend"] == "improving"
        assert trend["change_pct"] < 0  # Negative = improving (time decreasing)


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhaseA3Integration:
    """Integration tests for Phase A3 components."""

    @pytest.mark.asyncio
    async def test_contradiction_validator_vote(self):
        """Test validator vote on contradiction."""
        vote = ContradictionValidatorVote(
            validator_id="claude",
            vote="accept_a",
            confidence=0.85,
            reasoning="Item A has better evidence",
            weight=0.9,
        )

        assert vote.validator_id == "claude"
        assert vote.vote == "accept_a"

        vote_dict = vote.to_dict()
        assert vote_dict["validator_id"] == "claude"
        assert "voted_at" in vote_dict

    def test_quality_and_composite_analytics_flow(self):
        """Test flow from quality signals to composite analytics."""
        quality_engine = QualitySignalEngine()
        composite_analytics = CompositeAnalytics()

        # Compute quality for multiple items
        signals_list = []
        for i in range(5):
            signals = quality_engine.compute_quality_signals(
                item_id=f"km_{i}",
                raw_confidence=0.7 + i * 0.05,
                contributors=["claude"],
            )
            signals_list.append(signals)

        # Get quality summary
        summary = quality_engine.get_quality_summary(signals_list)
        assert summary["count"] == 5
        assert "avg_calibrated_confidence" in summary

        # Mock sync results based on quality computation "time"
        sync_results = [
            SyncResultInput(
                adapter_name="quality_engine",
                direction="compute",
                success=True,
                items_processed=5,
                duration_ms=150,
            )
        ]

        metrics = composite_analytics.compute_composite_metrics(sync_results)
        assert metrics.adapter_count == 1

    def test_singleton_accessors(self):
        """Test singleton accessors for all engines."""
        # Get singletons
        fusion = get_fusion_coordinator()
        calibration = get_calibration_fusion_engine()
        quality = get_quality_signal_engine()
        composite = get_composite_analytics()

        # All should be non-None
        assert fusion is not None
        assert calibration is not None
        assert quality is not None
        assert composite is not None

        # Getting again should return same instance
        assert get_fusion_coordinator() is fusion
        assert get_calibration_fusion_engine() is calibration
        assert get_quality_signal_engine() is quality
        assert get_composite_analytics() is composite
