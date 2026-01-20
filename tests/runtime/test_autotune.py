"""
Tests for runtime autotune module.

Tests cover:
- CostTier enum
- AutotuneConfig dataclass
- RunMetrics dataclass
- StopReason enum
- AutotuneDecision dataclass
- Autotuner class
- AutotunedDebateRunner class
"""

from datetime import datetime, timedelta
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.runtime.autotune import (
    AutotuneConfig,
    AutotuneDecision,
    AutotunedDebateRunner,
    Autotuner,
    CostTier,
    RunMetrics,
    StopReason,
)


class TestCostTier:
    """Tests for CostTier enum."""

    def test_has_all_tiers(self):
        """Enum has all expected tiers."""
        assert CostTier.FREE.value == "free"
        assert CostTier.CHEAP.value == "cheap"
        assert CostTier.STANDARD.value == "standard"
        assert CostTier.EXPENSIVE.value == "expensive"

    def test_tier_count(self):
        """Enum has exactly 4 tiers."""
        assert len(CostTier) == 4

    def test_tier_ordering_by_cost(self):
        """Tiers represent increasing cost levels."""
        tiers = [CostTier.FREE, CostTier.CHEAP, CostTier.STANDARD, CostTier.EXPENSIVE]
        values = ["free", "cheap", "standard", "expensive"]
        assert [t.value for t in tiers] == values


class TestAutotuneConfig:
    """Tests for AutotuneConfig dataclass."""

    def test_default_values(self):
        """Default values are sensible."""
        config = AutotuneConfig()

        assert config.max_cost_dollars == 1.0
        assert config.max_tokens == 100000
        assert config.max_rounds == 5
        assert config.max_duration_seconds == 300

    def test_early_stop_thresholds(self):
        """Early stop thresholds have sensible defaults."""
        config = AutotuneConfig()

        assert config.early_stop_support_variance == 0.1
        assert config.early_stop_verification_density == 0.7
        assert config.early_stop_consensus_confidence == 0.85

    def test_quality_targets(self):
        """Quality targets have sensible defaults."""
        config = AutotuneConfig()

        assert config.min_evidence_per_claim == 1
        assert config.min_rounds_before_stop == 1

    def test_cost_per_1k_tokens(self):
        """Cost weights are defined for all tiers."""
        config = AutotuneConfig()

        assert CostTier.FREE in config.cost_per_1k_tokens
        assert CostTier.CHEAP in config.cost_per_1k_tokens
        assert CostTier.STANDARD in config.cost_per_1k_tokens
        assert CostTier.EXPENSIVE in config.cost_per_1k_tokens

        # Verify ordering (free < cheap < standard < expensive)
        assert config.cost_per_1k_tokens[CostTier.FREE] == 0.0
        assert (
            config.cost_per_1k_tokens[CostTier.CHEAP] < config.cost_per_1k_tokens[CostTier.STANDARD]
        )
        assert (
            config.cost_per_1k_tokens[CostTier.STANDARD]
            < config.cost_per_1k_tokens[CostTier.EXPENSIVE]
        )

    def test_default_tier(self):
        """Default tier is STANDARD."""
        config = AutotuneConfig()
        assert config.default_tier == CostTier.STANDARD

    def test_custom_config(self):
        """Custom values can be set."""
        config = AutotuneConfig(
            max_cost_dollars=5.0,
            max_tokens=500000,
            max_rounds=10,
            max_duration_seconds=600,
            default_tier=CostTier.EXPENSIVE,
        )

        assert config.max_cost_dollars == 5.0
        assert config.max_tokens == 500000
        assert config.max_rounds == 10
        assert config.max_duration_seconds == 600
        assert config.default_tier == CostTier.EXPENSIVE


class TestRunMetrics:
    """Tests for RunMetrics dataclass."""

    def test_default_values(self):
        """Default values are zeros."""
        metrics = RunMetrics()

        assert metrics.rounds_completed == 0
        assert metrics.messages_sent == 0
        assert metrics.tokens_used == 0
        assert metrics.critiques_made == 0
        assert metrics.claims_made == 0
        assert metrics.evidence_cited == 0

    def test_quality_metrics_defaults(self):
        """Quality metrics have expected defaults."""
        metrics = RunMetrics()

        assert metrics.avg_support_score == 0.0
        assert metrics.support_score_variance == 1.0
        assert metrics.verification_density == 0.0
        assert metrics.consensus_confidence == 0.0

    def test_cost_metrics_defaults(self):
        """Cost metrics have expected defaults."""
        metrics = RunMetrics()

        assert metrics.estimated_cost == 0.0
        assert metrics.duration_seconds == 0.0

    def test_timestamp_defaults(self):
        """Timestamps default to None."""
        metrics = RunMetrics()

        assert metrics.started_at is None
        assert metrics.ended_at is None

    def test_round_metrics_default(self):
        """Round metrics default to empty list."""
        metrics = RunMetrics()
        assert metrics.round_metrics == []

    def test_add_round_metrics_basic(self):
        """add_round_metrics updates basic counters."""
        metrics = RunMetrics()

        metrics.add_round_metrics(
            round_num=0,
            tokens=1000,
            messages=5,
            support_scores=[0.8, 0.9, 0.7],
        )

        assert metrics.rounds_completed == 1
        assert metrics.tokens_used == 1000
        assert metrics.messages_sent == 5

    def test_add_round_metrics_cumulative(self):
        """add_round_metrics accumulates across rounds."""
        metrics = RunMetrics()

        metrics.add_round_metrics(0, 1000, 5, [0.8])
        metrics.add_round_metrics(1, 1500, 6, [0.85])

        assert metrics.rounds_completed == 2
        assert metrics.tokens_used == 2500
        assert metrics.messages_sent == 11

    def test_add_round_metrics_support_scores(self):
        """add_round_metrics calculates support score stats."""
        metrics = RunMetrics()

        scores = [0.8, 0.8, 0.8, 0.8]  # All same = variance 0
        metrics.add_round_metrics(0, 1000, 5, scores)

        assert metrics.avg_support_score == 0.8
        assert metrics.support_score_variance == 0.0

    def test_add_round_metrics_variance_calculation(self):
        """add_round_metrics calculates variance correctly."""
        metrics = RunMetrics()

        # Scores with known variance: [0.6, 0.8] => avg=0.7, var=0.01
        scores = [0.6, 0.8]
        metrics.add_round_metrics(0, 1000, 5, scores)

        assert metrics.avg_support_score == 0.7
        assert abs(metrics.support_score_variance - 0.01) < 0.001

    def test_add_round_metrics_empty_scores(self):
        """add_round_metrics handles empty scores list."""
        metrics = RunMetrics()

        metrics.add_round_metrics(0, 1000, 5, [])

        # Support scores should not change with empty list
        assert metrics.avg_support_score == 0.0
        assert metrics.support_score_variance == 1.0

    def test_add_round_metrics_records_history(self):
        """add_round_metrics appends to round_metrics."""
        metrics = RunMetrics()

        metrics.add_round_metrics(0, 1000, 5, [0.8])
        metrics.add_round_metrics(1, 1500, 6, [0.85])

        assert len(metrics.round_metrics) == 2
        assert metrics.round_metrics[0]["round"] == 0
        assert metrics.round_metrics[1]["round"] == 1
        assert metrics.round_metrics[0]["tokens"] == 1000
        assert metrics.round_metrics[1]["tokens"] == 1500

    def test_to_dict(self):
        """to_dict returns all metrics."""
        metrics = RunMetrics()
        metrics.rounds_completed = 2
        metrics.tokens_used = 5000
        metrics.estimated_cost = 0.15

        result = metrics.to_dict()

        assert result["rounds_completed"] == 2
        assert result["tokens_used"] == 5000
        assert result["estimated_cost"] == 0.15
        assert "round_metrics" in result


class TestStopReason:
    """Tests for StopReason enum."""

    def test_has_all_reasons(self):
        """Enum has all expected stop reasons."""
        assert StopReason.MAX_ROUNDS.value == "max_rounds"
        assert StopReason.MAX_COST.value == "max_cost"
        assert StopReason.MAX_TOKENS.value == "max_tokens"
        assert StopReason.MAX_DURATION.value == "max_duration"
        assert StopReason.CONSENSUS_REACHED.value == "consensus_reached"
        assert StopReason.QUALITY_THRESHOLD.value == "quality_threshold"
        assert StopReason.USER_REQUESTED.value == "user_requested"
        assert StopReason.ERROR.value == "error"

    def test_reason_count(self):
        """Enum has exactly 8 reasons."""
        assert len(StopReason) == 8


class TestAutotuneDecision:
    """Tests for AutotuneDecision dataclass."""

    def test_default_values(self):
        """Default values are sensible."""
        decision = AutotuneDecision(should_continue=True)

        assert decision.should_continue is True
        assert decision.stop_reason is None
        assert decision.recommended_tier == CostTier.STANDARD
        assert decision.suggested_action == ""
        assert decision.metrics_summary == {}

    def test_stop_decision(self):
        """Stop decisions have required fields."""
        decision = AutotuneDecision(
            should_continue=False,
            stop_reason=StopReason.MAX_ROUNDS,
            suggested_action="Debate ended after max rounds",
        )

        assert decision.should_continue is False
        assert decision.stop_reason == StopReason.MAX_ROUNDS

    def test_continue_with_tier_recommendation(self):
        """Continue decisions can recommend tier changes."""
        decision = AutotuneDecision(
            should_continue=True,
            recommended_tier=CostTier.CHEAP,
            suggested_action="Budget pressure - downgrading",
        )

        assert decision.should_continue is True
        assert decision.recommended_tier == CostTier.CHEAP


class TestAutotuner:
    """Tests for Autotuner class."""

    def test_init_default_config(self):
        """Initializes with default config."""
        tuner = Autotuner()

        assert tuner.config is not None
        assert isinstance(tuner.config, AutotuneConfig)
        assert tuner.metrics is not None
        assert tuner._current_tier == CostTier.STANDARD

    def test_init_custom_config(self):
        """Initializes with custom config."""
        config = AutotuneConfig(max_rounds=10, default_tier=CostTier.EXPENSIVE)
        tuner = Autotuner(config)

        assert tuner.config.max_rounds == 10
        assert tuner._current_tier == CostTier.EXPENSIVE

    def test_start(self):
        """start() records timestamp."""
        tuner = Autotuner()

        tuner.start()

        assert tuner._start_time is not None
        assert tuner.metrics.started_at is not None

    def test_end(self):
        """end() records duration."""
        tuner = Autotuner()
        tuner.start()

        tuner.end()

        assert tuner.metrics.ended_at is not None
        assert tuner.metrics.duration_seconds >= 0

    def test_end_without_start(self):
        """end() handles case where start() wasn't called."""
        tuner = Autotuner()

        # Should not raise
        tuner.end()

        assert tuner.metrics.ended_at is None

    def test_record_round(self):
        """record_round updates metrics."""
        tuner = Autotuner()
        tuner.start()

        tuner.record_round(
            round_num=0,
            tokens=5000,
            messages=10,
            support_scores=[0.8, 0.9],
            verified_claims=5,
            total_claims=10,
        )

        assert tuner.metrics.rounds_completed == 1
        assert tuner.metrics.tokens_used == 5000
        assert tuner.metrics.verification_density == 0.5
        assert tuner.metrics.claims_made == 10

    def test_record_round_cost_estimation(self):
        """record_round estimates cost based on tier."""
        config = AutotuneConfig(default_tier=CostTier.STANDARD)
        tuner = Autotuner(config)

        tuner.record_round(0, 10000, 5, [0.8])

        # Standard tier = $0.003 per 1K tokens
        expected_cost = 10 * 0.003
        assert abs(tuner.metrics.estimated_cost - expected_cost) < 0.001

    def test_record_consensus(self):
        """record_consensus updates confidence."""
        tuner = Autotuner()

        tuner.record_consensus(confidence=0.9, reached=True)

        assert tuner.metrics.consensus_confidence == 0.9

    def test_should_continue_max_rounds(self):
        """should_continue stops at max rounds."""
        config = AutotuneConfig(max_rounds=2)
        tuner = Autotuner(config)

        tuner.record_round(0, 1000, 5, [0.8])
        tuner.record_round(1, 1000, 5, [0.8])

        decision = tuner.should_continue()

        assert decision.should_continue is False
        assert decision.stop_reason == StopReason.MAX_ROUNDS

    def test_should_continue_max_cost(self):
        """should_continue stops when cost exceeded."""
        config = AutotuneConfig(max_cost_dollars=0.01, default_tier=CostTier.STANDARD)
        tuner = Autotuner(config)

        # 10K tokens at $0.003/1K = $0.03 > $0.01 budget
        tuner.record_round(0, 10000, 5, [0.8])

        decision = tuner.should_continue()

        assert decision.should_continue is False
        assert decision.stop_reason == StopReason.MAX_COST

    def test_should_continue_max_tokens(self):
        """should_continue stops when tokens exceeded."""
        config = AutotuneConfig(max_tokens=1000)
        tuner = Autotuner(config)

        tuner.record_round(0, 1500, 5, [0.8])

        decision = tuner.should_continue()

        assert decision.should_continue is False
        assert decision.stop_reason == StopReason.MAX_TOKENS

    def test_should_continue_max_duration(self):
        """should_continue stops when duration exceeded."""
        config = AutotuneConfig(max_duration_seconds=1)
        tuner = Autotuner(config)
        tuner.start()

        # Simulate time passing
        tuner._start_time = datetime.now() - timedelta(seconds=5)

        decision = tuner.should_continue()

        assert decision.should_continue is False
        assert decision.stop_reason == StopReason.MAX_DURATION

    def test_should_continue_consensus_reached(self):
        """should_continue stops when high consensus."""
        config = AutotuneConfig(
            early_stop_consensus_confidence=0.85,
            min_rounds_before_stop=1,
        )
        tuner = Autotuner(config)

        tuner.record_round(0, 1000, 5, [0.8])
        tuner.record_consensus(confidence=0.9, reached=True)

        decision = tuner.should_continue()

        assert decision.should_continue is False
        assert decision.stop_reason == StopReason.CONSENSUS_REACHED

    def test_should_continue_quality_threshold(self):
        """should_continue stops when quality thresholds met."""
        config = AutotuneConfig(
            early_stop_support_variance=0.1,
            early_stop_verification_density=0.7,
            min_rounds_before_stop=1,
        )
        tuner = Autotuner(config)

        # Low variance (all scores similar) and high verification
        tuner.record_round(0, 1000, 5, [0.8, 0.8, 0.8], verified_claims=8, total_claims=10)

        decision = tuner.should_continue()

        assert decision.should_continue is False
        assert decision.stop_reason == StopReason.QUALITY_THRESHOLD

    def test_should_continue_respects_min_rounds(self):
        """should_continue doesn't early stop before min rounds."""
        config = AutotuneConfig(
            early_stop_consensus_confidence=0.85,
            min_rounds_before_stop=2,
        )
        tuner = Autotuner(config)

        # High consensus but only 1 round completed
        tuner.record_round(0, 1000, 5, [0.8])
        tuner.record_consensus(confidence=0.95, reached=True)

        decision = tuner.should_continue()

        # Should continue because min rounds not met
        assert decision.should_continue is True

    def test_should_continue_recommends_tier(self):
        """should_continue recommends tier based on budget."""
        config = AutotuneConfig(max_cost_dollars=1.0, default_tier=CostTier.STANDARD)
        tuner = Autotuner(config)

        # Use 10% of budget - should recommend expensive
        tuner.record_round(0, 10000, 5, [0.8])
        tuner.metrics.estimated_cost = 0.1

        decision = tuner.should_continue()

        assert decision.should_continue is True
        assert decision.recommended_tier == CostTier.EXPENSIVE

    def test_recommend_tier_budget_pressure(self):
        """_recommend_tier downgrades under budget pressure."""
        tuner = Autotuner()

        assert tuner._recommend_tier(0.1) == CostTier.EXPENSIVE  # < 30%
        assert tuner._recommend_tier(0.4) == CostTier.STANDARD  # 30-60%
        assert tuner._recommend_tier(0.7) == CostTier.CHEAP  # 60-85%
        assert tuner._recommend_tier(0.9) == CostTier.FREE  # > 85%

    def test_get_budget_remaining(self):
        """get_budget_remaining returns correct values."""
        config = AutotuneConfig(
            max_cost_dollars=1.0,
            max_tokens=100000,
            max_rounds=5,
        )
        tuner = Autotuner(config)

        tuner.record_round(0, 20000, 5, [0.8])
        tuner.metrics.estimated_cost = 0.2

        budget = tuner.get_budget_remaining()

        assert budget["cost_remaining"] == 0.8
        assert budget["tokens_remaining"] == 80000
        assert budget["rounds_remaining"] == 4
        assert budget["budget_used_percent"] == 20.0

    def test_suggest_rounds(self):
        """suggest_rounds estimates based on history."""
        config = AutotuneConfig(max_tokens=100000, max_rounds=5)
        tuner = Autotuner(config)

        # After 1 round using 20K tokens
        tuner.record_round(0, 20000, 5, [0.8])

        suggested = tuner.suggest_rounds()

        # 80K remaining / 20K per round = 4, but capped at 3
        assert suggested <= 3

    def test_suggest_rounds_no_history(self):
        """suggest_rounds uses default without history."""
        config = AutotuneConfig(max_tokens=50000, max_rounds=5)
        tuner = Autotuner(config)

        suggested = tuner.suggest_rounds()

        # Uses 5K default estimate: 50K / 5K = 10, capped at min(4 remaining, 3)
        assert suggested <= 3


class TestAutotunedDebateRunner:
    """Tests for AutotunedDebateRunner class."""

    def test_init(self):
        """Initializes with arena and config."""
        arena = MagicMock()
        config = AutotuneConfig(max_rounds=3)

        runner = AutotunedDebateRunner(arena, config)

        assert runner.arena is arena
        assert runner.tuner.config.max_rounds == 3

    def test_init_default_config(self):
        """Initializes with default config when none provided."""
        arena = MagicMock()

        runner = AutotunedDebateRunner(arena)

        assert runner.tuner.config is not None

    @pytest.mark.asyncio
    async def test_run_calls_arena(self):
        """run() executes arena.run()."""
        arena = MagicMock()
        arena.run = AsyncMock(return_value="debate_result")

        runner = AutotunedDebateRunner(arena)
        result, metrics = await runner.run()

        arena.run.assert_called_once()
        assert result == "debate_result"
        assert isinstance(metrics, RunMetrics)

    @pytest.mark.asyncio
    async def test_run_starts_and_ends_tuner(self):
        """run() properly starts and ends tuner."""
        arena = MagicMock()
        arena.run = AsyncMock(return_value="result")

        runner = AutotunedDebateRunner(arena)

        with patch.object(runner.tuner, "start") as mock_start:
            with patch.object(runner.tuner, "end") as mock_end:
                await runner.run()

                mock_start.assert_called_once()
                mock_end.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_ends_tuner_on_error(self):
        """run() ends tuner even when arena raises."""
        arena = MagicMock()
        arena.run = AsyncMock(side_effect=RuntimeError("Arena failed"))

        runner = AutotunedDebateRunner(arena)

        with patch.object(runner.tuner, "end") as mock_end:
            with pytest.raises(RuntimeError):
                await runner.run()

            # end() should still be called
            mock_end.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_callback(self):
        """run() invokes user callback."""
        arena = MagicMock()
        arena.run = AsyncMock(return_value="result")

        callback = MagicMock()
        runner = AutotunedDebateRunner(arena)

        await runner.run(on_round_complete=callback)

        # Callback passed through to arena
        arena.run.assert_called_once()

    def test_on_round_records_metrics(self):
        """_on_round records metrics from stats."""
        arena = MagicMock()
        runner = AutotunedDebateRunner(arena)

        stats = {
            "tokens": 5000,
            "messages": 10,
            "support_scores": [0.8, 0.9],
            "verified_claims": 3,
            "total_claims": 5,
        }

        runner._on_round(round_num=0, stats=stats, user_callback=None)

        assert runner.tuner.metrics.tokens_used == 5000
        assert runner.tuner.metrics.messages_sent == 10

    def test_on_round_calls_user_callback(self):
        """_on_round invokes user callback."""
        arena = MagicMock()
        runner = AutotunedDebateRunner(arena)

        callback = MagicMock()
        stats = {"tokens": 1000, "messages": 5}

        runner._on_round(round_num=0, stats=stats, user_callback=callback)

        callback.assert_called_once_with(0, runner.tuner.metrics)

    def test_on_round_returns_decision(self):
        """_on_round returns tuner decision."""
        arena = MagicMock()
        runner = AutotunedDebateRunner(arena)

        stats = {"tokens": 1000, "messages": 5}

        decision = runner._on_round(round_num=0, stats=stats, user_callback=None)

        assert isinstance(decision, AutotuneDecision)


class TestAutotunerIntegration:
    """Integration tests for autotuner workflows."""

    def test_full_debate_flow(self):
        """Simulates a full debate with autotuning."""
        config = AutotuneConfig(
            max_rounds=5,
            max_cost_dollars=1.0,
            early_stop_consensus_confidence=0.9,
            min_rounds_before_stop=2,
        )
        tuner = Autotuner(config)

        tuner.start()

        # Round 1
        tuner.record_round(0, 5000, 5, [0.7, 0.6, 0.8])
        decision = tuner.should_continue()
        assert decision.should_continue is True

        # Round 2
        tuner.record_round(1, 4000, 6, [0.75, 0.78, 0.76])
        decision = tuner.should_continue()
        assert decision.should_continue is True

        # Round 3 - consensus reached
        tuner.record_round(2, 3500, 4, [0.85, 0.88, 0.86])
        tuner.record_consensus(confidence=0.92, reached=True)
        decision = tuner.should_continue()

        assert decision.should_continue is False
        assert decision.stop_reason == StopReason.CONSENSUS_REACHED

        tuner.end()

        # Verify final metrics
        assert tuner.metrics.rounds_completed == 3
        assert tuner.metrics.tokens_used == 12500
        assert tuner.metrics.duration_seconds >= 0

    def test_budget_pressure_downgrade(self):
        """Verifies tier recommendation changes with budget pressure."""
        config = AutotuneConfig(
            max_cost_dollars=0.10,
            max_rounds=10,
            default_tier=CostTier.EXPENSIVE,
        )
        tuner = Autotuner(config)

        # Start with expensive tier
        assert tuner._current_tier == CostTier.EXPENSIVE

        # Round 1 - expensive, 40% budget used
        tuner.record_round(0, 5000, 5, [0.8])
        tuner.metrics.estimated_cost = 0.04
        decision = tuner.should_continue()
        assert decision.recommended_tier == CostTier.STANDARD  # 40% = standard

        # Round 2 - 70% budget used
        tuner.record_round(1, 5000, 5, [0.8])
        tuner.metrics.estimated_cost = 0.07
        decision = tuner.should_continue()
        assert decision.recommended_tier == CostTier.CHEAP  # 70% = cheap

        # Round 3 - 90% budget used
        tuner.record_round(2, 5000, 5, [0.8])
        tuner.metrics.estimated_cost = 0.09
        decision = tuner.should_continue()
        assert decision.recommended_tier == CostTier.FREE  # 90% = free

    def test_metrics_serialization(self):
        """Verifies metrics can be serialized."""
        tuner = Autotuner()
        tuner.start()

        tuner.record_round(0, 5000, 10, [0.8, 0.9])
        tuner.record_consensus(0.85, True)

        tuner.end()

        metrics_dict = tuner.metrics.to_dict()

        # Verify serializable
        import json

        serialized = json.dumps(metrics_dict)
        assert isinstance(serialized, str)

        # Verify content
        parsed = json.loads(serialized)
        assert parsed["rounds_completed"] == 1
        assert parsed["tokens_used"] == 5000
