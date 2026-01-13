"""
Tests for the Autotuner module.

Tests budget-aware debate optimization including cost tracking,
early stopping, and model tier recommendations.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from aragora.runtime.autotune import (
    CostTier,
    AutotuneConfig,
    RunMetrics,
    StopReason,
    AutotuneDecision,
    Autotuner,
    AutotunedDebateRunner,
)


# =============================================================================
# CostTier Tests
# =============================================================================


class TestCostTier:
    """Tests for CostTier enum."""

    def test_cost_tier_values(self):
        """Test cost tier enum values."""
        assert CostTier.FREE.value == "free"
        assert CostTier.CHEAP.value == "cheap"
        assert CostTier.STANDARD.value == "standard"
        assert CostTier.EXPENSIVE.value == "expensive"

    def test_cost_tier_ordering(self):
        """Test all tiers exist."""
        tiers = list(CostTier)
        assert len(tiers) == 4


# =============================================================================
# AutotuneConfig Tests
# =============================================================================


class TestAutotuneConfig:
    """Tests for AutotuneConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AutotuneConfig()
        assert config.max_cost_dollars == 1.0
        assert config.max_tokens == 100000
        assert config.max_rounds == 5
        assert config.max_duration_seconds == 300
        assert config.min_rounds_before_stop == 1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AutotuneConfig(
            max_cost_dollars=5.0,
            max_tokens=500000,
            max_rounds=10,
        )
        assert config.max_cost_dollars == 5.0
        assert config.max_tokens == 500000
        assert config.max_rounds == 10

    def test_early_stop_thresholds(self):
        """Test early stop threshold defaults."""
        config = AutotuneConfig()
        assert config.early_stop_support_variance == 0.1
        assert config.early_stop_verification_density == 0.7
        assert config.early_stop_consensus_confidence == 0.85

    def test_cost_per_1k_tokens(self):
        """Test cost per 1k tokens by tier."""
        config = AutotuneConfig()
        assert config.cost_per_1k_tokens[CostTier.FREE] == 0.0
        assert config.cost_per_1k_tokens[CostTier.CHEAP] == 0.0005
        assert config.cost_per_1k_tokens[CostTier.STANDARD] == 0.003
        assert config.cost_per_1k_tokens[CostTier.EXPENSIVE] == 0.03


# =============================================================================
# RunMetrics Tests
# =============================================================================


class TestRunMetrics:
    """Tests for RunMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = RunMetrics()
        assert metrics.rounds_completed == 0
        assert metrics.messages_sent == 0
        assert metrics.tokens_used == 0
        assert metrics.avg_support_score == 0.0
        assert metrics.support_score_variance == 1.0
        assert metrics.verification_density == 0.0

    def test_add_round_metrics(self):
        """Test recording round metrics."""
        metrics = RunMetrics()
        metrics.add_round_metrics(
            round_num=0,
            tokens=1000,
            messages=5,
            support_scores=[0.8, 0.7, 0.9],
        )

        assert metrics.rounds_completed == 1
        assert metrics.tokens_used == 1000
        assert metrics.messages_sent == 5
        assert 0.79 < metrics.avg_support_score < 0.81  # ~0.8

    def test_add_multiple_rounds(self):
        """Test recording multiple rounds."""
        metrics = RunMetrics()

        metrics.add_round_metrics(0, 1000, 5, [0.5, 0.6])
        metrics.add_round_metrics(1, 1500, 4, [0.7, 0.8])

        assert metrics.rounds_completed == 2
        assert metrics.tokens_used == 2500
        assert metrics.messages_sent == 9

    def test_support_score_variance_calculation(self):
        """Test variance calculation for support scores."""
        metrics = RunMetrics()
        # All same score = 0 variance
        metrics.add_round_metrics(0, 1000, 5, [0.5, 0.5, 0.5])
        assert metrics.support_score_variance == 0.0

    def test_to_dict(self):
        """Test metrics serialization."""
        metrics = RunMetrics()
        metrics.add_round_metrics(0, 1000, 5, [0.8])

        data = metrics.to_dict()
        assert "rounds_completed" in data
        assert "tokens_used" in data
        assert "round_metrics" in data
        assert len(data["round_metrics"]) == 1

    def test_round_metrics_history(self):
        """Test that round metrics are stored in history."""
        metrics = RunMetrics()
        metrics.add_round_metrics(0, 1000, 5, [0.7])
        metrics.add_round_metrics(1, 1200, 6, [0.8])

        assert len(metrics.round_metrics) == 2
        assert metrics.round_metrics[0]["round"] == 0
        assert metrics.round_metrics[1]["round"] == 1


# =============================================================================
# StopReason Tests
# =============================================================================


class TestStopReason:
    """Tests for StopReason enum."""

    def test_stop_reason_values(self):
        """Test all stop reasons exist."""
        assert StopReason.MAX_ROUNDS.value == "max_rounds"
        assert StopReason.MAX_COST.value == "max_cost"
        assert StopReason.MAX_TOKENS.value == "max_tokens"
        assert StopReason.MAX_DURATION.value == "max_duration"
        assert StopReason.CONSENSUS_REACHED.value == "consensus_reached"
        assert StopReason.QUALITY_THRESHOLD.value == "quality_threshold"


# =============================================================================
# Autotuner Tests
# =============================================================================


class TestAutotuner:
    """Tests for Autotuner class."""

    def test_autotuner_creation(self):
        """Test basic autotuner creation."""
        tuner = Autotuner()
        assert tuner.config is not None
        assert tuner.metrics is not None

    def test_autotuner_with_custom_config(self):
        """Test autotuner with custom config."""
        config = AutotuneConfig(max_rounds=10, max_cost_dollars=5.0)
        tuner = Autotuner(config)
        assert tuner.config.max_rounds == 10
        assert tuner.config.max_cost_dollars == 5.0

    def test_start_and_end(self):
        """Test start and end tracking."""
        tuner = Autotuner()
        tuner.start()

        assert tuner._start_time is not None
        assert tuner.metrics.started_at is not None

        tuner.end()
        assert tuner.metrics.ended_at is not None
        assert tuner.metrics.duration_seconds >= 0

    def test_record_round(self):
        """Test recording round data."""
        tuner = Autotuner()
        tuner.start()

        tuner.record_round(
            round_num=0,
            tokens=1000,
            messages=5,
            support_scores=[0.7, 0.8],
            verified_claims=3,
            total_claims=5,
        )

        assert tuner.metrics.rounds_completed == 1
        assert tuner.metrics.tokens_used == 1000
        assert tuner.metrics.verification_density == 0.6
        assert tuner.metrics.estimated_cost > 0

    def test_record_consensus(self):
        """Test recording consensus."""
        tuner = Autotuner()
        tuner.record_consensus(confidence=0.9, reached=True)
        assert tuner.metrics.consensus_confidence == 0.9

    def test_should_continue_max_rounds(self):
        """Test stopping at max rounds."""
        config = AutotuneConfig(max_rounds=2)
        tuner = Autotuner(config)

        # Simulate 2 rounds
        tuner.record_round(0, 1000, 5, [0.5])
        tuner.record_round(1, 1000, 5, [0.5])

        decision = tuner.should_continue()
        assert decision.should_continue is False
        assert decision.stop_reason == StopReason.MAX_ROUNDS

    def test_should_continue_max_tokens(self):
        """Test stopping at max tokens."""
        config = AutotuneConfig(max_tokens=5000)
        tuner = Autotuner(config)

        # Use up tokens
        tuner.record_round(0, 6000, 10, [0.5])

        decision = tuner.should_continue()
        assert decision.should_continue is False
        assert decision.stop_reason == StopReason.MAX_TOKENS

    def test_should_continue_max_cost(self):
        """Test stopping at max cost."""
        config = AutotuneConfig(max_cost_dollars=0.001)
        tuner = Autotuner(config)

        # Run up cost (standard tier = $0.003/1k tokens)
        tuner.record_round(0, 1000, 5, [0.5])  # ~$0.003

        decision = tuner.should_continue()
        assert decision.should_continue is False
        assert decision.stop_reason == StopReason.MAX_COST

    def test_should_continue_consensus_reached(self):
        """Test stopping when consensus is reached."""
        config = AutotuneConfig(early_stop_consensus_confidence=0.85)
        tuner = Autotuner(config)

        tuner.record_round(0, 1000, 5, [0.9])
        tuner.record_consensus(confidence=0.9, reached=True)

        decision = tuner.should_continue()
        assert decision.should_continue is False
        assert decision.stop_reason == StopReason.CONSENSUS_REACHED

    def test_should_continue_quality_threshold(self):
        """Test stopping when quality threshold is met."""
        config = AutotuneConfig(
            early_stop_support_variance=0.1,
            early_stop_verification_density=0.7,
        )
        tuner = Autotuner(config)

        # Low variance, high verification density
        tuner.record_round(0, 1000, 5, [0.8, 0.8, 0.8])  # variance = 0
        tuner.record_round(
            1,
            1000,
            5,
            [0.8, 0.8],
            verified_claims=8,
            total_claims=10,  # 0.8 density
        )

        decision = tuner.should_continue()
        assert decision.should_continue is False
        assert decision.stop_reason == StopReason.QUALITY_THRESHOLD

    def test_should_continue_true(self):
        """Test continuing when no stop conditions met."""
        tuner = Autotuner()
        tuner.start()
        tuner.record_round(0, 100, 2, [0.5])

        decision = tuner.should_continue()
        assert decision.should_continue is True
        assert decision.stop_reason is None

    def test_recommend_tier_low_budget(self):
        """Test tier recommendation with low budget usage."""
        tuner = Autotuner()
        tier = tuner._recommend_tier(0.1)
        assert tier == CostTier.EXPENSIVE

    def test_recommend_tier_medium_budget(self):
        """Test tier recommendation with medium budget usage."""
        tuner = Autotuner()
        tier = tuner._recommend_tier(0.5)
        assert tier == CostTier.STANDARD

    def test_recommend_tier_high_budget(self):
        """Test tier recommendation with high budget usage."""
        tuner = Autotuner()
        tier = tuner._recommend_tier(0.75)
        assert tier == CostTier.CHEAP

    def test_recommend_tier_critical_budget(self):
        """Test tier recommendation with critical budget usage."""
        tuner = Autotuner()
        tier = tuner._recommend_tier(0.9)
        assert tier == CostTier.FREE

    def test_get_budget_remaining(self):
        """Test budget remaining calculation."""
        config = AutotuneConfig(
            max_cost_dollars=1.0,
            max_tokens=10000,
            max_rounds=5,
        )
        tuner = Autotuner(config)
        tuner.record_round(0, 2000, 5, [0.5])

        budget = tuner.get_budget_remaining()
        assert budget["tokens_remaining"] == 8000
        assert budget["rounds_remaining"] == 4
        assert budget["budget_used_percent"] > 0

    def test_suggest_rounds(self):
        """Test round suggestion."""
        tuner = Autotuner()
        tuner.record_round(0, 5000, 5, [0.5])  # 5000 tokens/round

        suggested = tuner.suggest_rounds()
        assert suggested > 0
        assert suggested <= 3  # Max 3 suggested

    def test_suggest_rounds_no_history(self):
        """Test round suggestion without history."""
        tuner = Autotuner()
        suggested = tuner.suggest_rounds()
        assert suggested > 0


# =============================================================================
# AutotunedDebateRunner Tests
# =============================================================================


class TestAutotunedDebateRunner:
    """Tests for AutotunedDebateRunner."""

    def test_runner_creation(self):
        """Test runner creation."""
        mock_arena = MagicMock()
        runner = AutotunedDebateRunner(mock_arena)
        assert runner.arena is mock_arena
        assert runner.tuner is not None

    def test_runner_with_config(self):
        """Test runner with custom config."""
        mock_arena = MagicMock()
        config = AutotuneConfig(max_rounds=10)
        runner = AutotunedDebateRunner(mock_arena, config)
        assert runner.tuner.config.max_rounds == 10

    @pytest.mark.asyncio
    async def test_run_calls_arena(self):
        """Test that run calls arena.run."""
        mock_arena = MagicMock()
        mock_arena.run = AsyncMock(return_value={"status": "completed"})

        runner = AutotunedDebateRunner(mock_arena)
        result, metrics = await runner.run()

        mock_arena.run.assert_called_once()
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_run_tracks_metrics(self):
        """Test that run tracks metrics."""
        mock_arena = MagicMock()
        mock_arena.run = AsyncMock(return_value={"status": "completed"})

        runner = AutotunedDebateRunner(mock_arena)
        result, metrics = await runner.run()

        assert metrics.started_at is not None
        assert metrics.ended_at is not None

    def test_on_round_records_metrics(self):
        """Test round callback records metrics."""
        mock_arena = MagicMock()
        runner = AutotunedDebateRunner(mock_arena)

        stats = {
            "tokens": 1000,
            "messages": 5,
            "support_scores": [0.7, 0.8],
            "verified_claims": 3,
            "total_claims": 5,
        }

        runner._on_round(round_num=0, stats=stats, user_callback=None)

        assert runner.tuner.metrics.rounds_completed == 1
        assert runner.tuner.metrics.tokens_used == 1000

    def test_on_round_calls_user_callback(self):
        """Test round callback calls user callback."""
        mock_arena = MagicMock()
        runner = AutotunedDebateRunner(mock_arena)

        callback = MagicMock()
        stats = {"tokens": 1000, "messages": 5, "support_scores": []}

        runner._on_round(round_num=0, stats=stats, user_callback=callback)

        callback.assert_called_once()


# =============================================================================
# Integration Tests
# =============================================================================


class TestAutotuneIntegration:
    """Integration tests for autotuning."""

    def test_full_debate_simulation(self):
        """Test simulating a full autotuned debate."""
        config = AutotuneConfig(
            max_rounds=3,
            max_cost_dollars=1.0,
        )
        tuner = Autotuner(config)
        tuner.start()

        # Round 1
        tuner.record_round(0, 2000, 5, [0.5, 0.6, 0.7])
        decision = tuner.should_continue()
        assert decision.should_continue is True

        # Round 2
        tuner.record_round(1, 2500, 6, [0.65, 0.7, 0.75])
        decision = tuner.should_continue()
        assert decision.should_continue is True

        # Round 3 - should stop at max rounds
        tuner.record_round(2, 3000, 7, [0.7, 0.75, 0.8])
        decision = tuner.should_continue()
        assert decision.should_continue is False
        assert decision.stop_reason == StopReason.MAX_ROUNDS

        tuner.end()

        # Verify final metrics
        assert tuner.metrics.rounds_completed == 3
        assert tuner.metrics.tokens_used == 7500

    def test_early_consensus_stop(self):
        """Test early stopping on consensus."""
        config = AutotuneConfig(
            max_rounds=10,
            early_stop_consensus_confidence=0.85,
        )
        tuner = Autotuner(config)
        tuner.start()

        tuner.record_round(0, 2000, 5, [0.7, 0.8])

        # High consensus early
        tuner.record_consensus(confidence=0.9, reached=True)

        decision = tuner.should_continue()
        assert decision.should_continue is False
        assert decision.stop_reason == StopReason.CONSENSUS_REACHED

    def test_budget_pressure_downgrade(self):
        """Test tier downgrade under budget pressure."""
        config = AutotuneConfig(
            max_cost_dollars=0.01,
            max_rounds=10,
        )
        tuner = Autotuner(config)
        tuner.start()

        # Use 70% of budget (at standard tier: $0.003/1k)
        # $0.007 = 2333 tokens at standard
        tuner.record_round(0, 2500, 5, [0.5])

        decision = tuner.should_continue()
        # Should recommend cheaper tier due to budget pressure
        assert decision.recommended_tier in [CostTier.CHEAP, CostTier.FREE]
