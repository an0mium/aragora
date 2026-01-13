"""
Tests for aragora/runtime/autotune.py - Budget-aware debate optimization.

This module is SECURITY-CRITICAL for financial controls:
- Budget enforcement prevents overspend
- Early stop conditions optimize costs
- Model tier selection controls API costs
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from aragora.runtime.autotune import (
    AutotuneConfig,
    AutotuneDecision,
    AutotunedDebateRunner,
    Autotuner,
    CostTier,
    RunMetrics,
    StopReason,
)


class TestCostTier(unittest.TestCase):
    """Tests for CostTier enum."""

    def test_all_tiers_defined(self):
        """All expected cost tiers should be defined."""
        self.assertEqual(CostTier.FREE.value, "free")
        self.assertEqual(CostTier.CHEAP.value, "cheap")
        self.assertEqual(CostTier.STANDARD.value, "standard")
        self.assertEqual(CostTier.EXPENSIVE.value, "expensive")

    def test_tier_count(self):
        """Should have exactly 4 cost tiers."""
        self.assertEqual(len(CostTier), 4)


class TestAutotuneConfig(unittest.TestCase):
    """Tests for AutotuneConfig defaults and customization."""

    def test_default_values(self):
        """Default config should have sensible values."""
        config = AutotuneConfig()
        self.assertEqual(config.max_cost_dollars, 1.0)
        self.assertEqual(config.max_tokens, 100000)
        self.assertEqual(config.max_rounds, 5)
        self.assertEqual(config.max_duration_seconds, 300)

    def test_early_stop_defaults(self):
        """Early stop thresholds should be set."""
        config = AutotuneConfig()
        self.assertEqual(config.early_stop_support_variance, 0.1)
        self.assertEqual(config.early_stop_verification_density, 0.7)
        self.assertEqual(config.early_stop_consensus_confidence, 0.85)

    def test_cost_per_1k_tokens(self):
        """Cost rates should be defined for all tiers."""
        config = AutotuneConfig()
        self.assertEqual(config.cost_per_1k_tokens[CostTier.FREE], 0.0)
        self.assertEqual(config.cost_per_1k_tokens[CostTier.CHEAP], 0.0005)
        self.assertEqual(config.cost_per_1k_tokens[CostTier.STANDARD], 0.003)
        self.assertEqual(config.cost_per_1k_tokens[CostTier.EXPENSIVE], 0.03)

    def test_custom_config(self):
        """Should allow custom configuration."""
        config = AutotuneConfig(
            max_cost_dollars=5.0,
            max_tokens=500000,
            max_rounds=10,
        )
        self.assertEqual(config.max_cost_dollars, 5.0)
        self.assertEqual(config.max_tokens, 500000)
        self.assertEqual(config.max_rounds, 10)


class TestRunMetrics(unittest.TestCase):
    """Tests for RunMetrics tracking."""

    def test_initial_state(self):
        """New metrics should start at zero."""
        metrics = RunMetrics()
        self.assertEqual(metrics.rounds_completed, 0)
        self.assertEqual(metrics.messages_sent, 0)
        self.assertEqual(metrics.tokens_used, 0)
        self.assertEqual(metrics.estimated_cost, 0.0)

    def test_add_round_metrics_accumulates_tokens(self):
        """Adding round metrics should accumulate tokens."""
        metrics = RunMetrics()
        metrics.add_round_metrics(0, tokens=1000, messages=5, support_scores=[0.8])
        metrics.add_round_metrics(1, tokens=1500, messages=6, support_scores=[0.85])

        self.assertEqual(metrics.tokens_used, 2500)
        self.assertEqual(metrics.messages_sent, 11)
        self.assertEqual(metrics.rounds_completed, 2)

    def test_add_round_metrics_updates_support_score(self):
        """Should calculate average and variance of support scores."""
        metrics = RunMetrics()
        # Scores with known variance
        metrics.add_round_metrics(0, tokens=1000, messages=5, support_scores=[0.7, 0.8, 0.9])

        # Average: (0.7 + 0.8 + 0.9) / 3 = 0.8
        self.assertAlmostEqual(metrics.avg_support_score, 0.8, places=5)
        # Variance: ((0.1)^2 + 0 + (0.1)^2) / 3 = 0.02/3 ≈ 0.00667
        self.assertAlmostEqual(metrics.support_score_variance, 0.00667, places=4)

    def test_add_round_metrics_empty_scores(self):
        """Should handle empty support scores gracefully."""
        metrics = RunMetrics()
        metrics.add_round_metrics(0, tokens=1000, messages=5, support_scores=[])
        # Should not crash, values remain at default
        self.assertEqual(metrics.avg_support_score, 0.0)

    def test_round_metrics_history(self):
        """Should maintain history of per-round metrics."""
        metrics = RunMetrics()
        metrics.add_round_metrics(0, tokens=1000, messages=5, support_scores=[0.8])
        metrics.add_round_metrics(1, tokens=1500, messages=6, support_scores=[0.85])

        self.assertEqual(len(metrics.round_metrics), 2)
        self.assertEqual(metrics.round_metrics[0]["round"], 0)
        self.assertEqual(metrics.round_metrics[0]["tokens"], 1000)
        self.assertEqual(metrics.round_metrics[1]["round"], 1)
        self.assertEqual(metrics.round_metrics[1]["tokens"], 1500)

    def test_to_dict_serialization(self):
        """Should serialize all metrics to dictionary."""
        metrics = RunMetrics()
        metrics.add_round_metrics(0, tokens=1000, messages=5, support_scores=[0.8])

        d = metrics.to_dict()
        self.assertIn("rounds_completed", d)
        self.assertIn("tokens_used", d)
        self.assertIn("estimated_cost", d)
        self.assertIn("round_metrics", d)
        self.assertEqual(d["tokens_used"], 1000)


class TestStopReason(unittest.TestCase):
    """Tests for StopReason enum."""

    def test_all_stop_reasons_defined(self):
        """All expected stop reasons should be defined."""
        self.assertEqual(StopReason.MAX_ROUNDS.value, "max_rounds")
        self.assertEqual(StopReason.MAX_COST.value, "max_cost")
        self.assertEqual(StopReason.MAX_TOKENS.value, "max_tokens")
        self.assertEqual(StopReason.MAX_DURATION.value, "max_duration")
        self.assertEqual(StopReason.CONSENSUS_REACHED.value, "consensus_reached")
        self.assertEqual(StopReason.QUALITY_THRESHOLD.value, "quality_threshold")
        self.assertEqual(StopReason.USER_REQUESTED.value, "user_requested")
        self.assertEqual(StopReason.ERROR.value, "error")


class TestAutotunerBudgetEnforcement(unittest.TestCase):
    """Tests for budget limit enforcement - CRITICAL for financial controls."""

    def test_stops_at_max_rounds(self):
        """Should stop when max rounds reached."""
        config = AutotuneConfig(max_rounds=3)
        tuner = Autotuner(config)
        tuner.start()

        # Complete 3 rounds
        for i in range(3):
            tuner.record_round(i, tokens=1000, messages=5, support_scores=[0.5])

        decision = tuner.should_continue()
        self.assertFalse(decision.should_continue)
        self.assertEqual(decision.stop_reason, StopReason.MAX_ROUNDS)

    def test_stops_at_max_cost(self):
        """Should stop when cost budget exceeded."""
        config = AutotuneConfig(max_cost_dollars=0.01)  # Very low budget
        tuner = Autotuner(config)
        tuner.start()

        # Use enough tokens to exceed budget
        # STANDARD tier: $0.003 per 1K tokens
        # 4000 tokens = $0.012 > $0.01 budget
        tuner.record_round(0, tokens=4000, messages=5, support_scores=[0.5])

        decision = tuner.should_continue()
        self.assertFalse(decision.should_continue)
        self.assertEqual(decision.stop_reason, StopReason.MAX_COST)

    def test_stops_at_max_tokens(self):
        """Should stop when token limit exceeded."""
        config = AutotuneConfig(max_tokens=5000)
        tuner = Autotuner(config)
        tuner.start()

        # Use more than max tokens
        tuner.record_round(0, tokens=6000, messages=5, support_scores=[0.5])

        decision = tuner.should_continue()
        self.assertFalse(decision.should_continue)
        self.assertEqual(decision.stop_reason, StopReason.MAX_TOKENS)

    def test_stops_at_max_duration(self):
        """Should stop when duration limit exceeded."""
        config = AutotuneConfig(max_duration_seconds=60)
        tuner = Autotuner(config)
        tuner.start()

        # Simulate time passage
        with patch.object(tuner, "_start_time", datetime.now() - timedelta(seconds=61)):
            decision = tuner.should_continue()

        self.assertFalse(decision.should_continue)
        self.assertEqual(decision.stop_reason, StopReason.MAX_DURATION)

    def test_continues_within_budget(self):
        """Should continue when within all limits."""
        config = AutotuneConfig(
            max_rounds=5,
            max_cost_dollars=1.0,
            max_tokens=100000,
        )
        tuner = Autotuner(config)
        tuner.start()

        # Use modest resources
        tuner.record_round(0, tokens=1000, messages=5, support_scores=[0.5])

        decision = tuner.should_continue()
        self.assertTrue(decision.should_continue)
        self.assertIsNone(decision.stop_reason)


class TestAutotunerEarlyStop(unittest.TestCase):
    """Tests for early-stop conditions based on quality metrics."""

    def test_stops_on_high_consensus_confidence(self):
        """Should stop when consensus confidence exceeds threshold."""
        config = AutotuneConfig(
            early_stop_consensus_confidence=0.85,
            min_rounds_before_stop=1,
        )
        tuner = Autotuner(config)
        tuner.start()

        # Complete min rounds
        tuner.record_round(0, tokens=1000, messages=5, support_scores=[0.8])
        # Record high consensus
        tuner.record_consensus(confidence=0.90, reached=True)

        decision = tuner.should_continue()
        self.assertFalse(decision.should_continue)
        self.assertEqual(decision.stop_reason, StopReason.CONSENSUS_REACHED)

    def test_no_early_stop_before_min_rounds(self):
        """Should not early-stop before minimum rounds completed."""
        config = AutotuneConfig(
            early_stop_consensus_confidence=0.85,
            min_rounds_before_stop=2,
        )
        tuner = Autotuner(config)
        tuner.start()

        # Only 1 round (less than min_rounds_before_stop)
        tuner.record_round(0, tokens=1000, messages=5, support_scores=[0.8])
        tuner.record_consensus(confidence=0.95, reached=True)  # High confidence

        decision = tuner.should_continue()
        self.assertTrue(decision.should_continue)  # Should continue despite high confidence

    def test_stops_on_quality_threshold(self):
        """Should stop when quality metrics indicate convergence."""
        config = AutotuneConfig(
            early_stop_support_variance=0.1,
            early_stop_verification_density=0.7,
            min_rounds_before_stop=1,
        )
        tuner = Autotuner(config)
        tuner.start()

        # Complete min rounds with low variance
        tuner.record_round(
            0,
            tokens=1000,
            messages=5,
            support_scores=[0.8, 0.81, 0.79],  # Very tight spread = low variance
            verified_claims=8,
            total_claims=10,  # 80% verification density
        )

        decision = tuner.should_continue()
        self.assertFalse(decision.should_continue)
        self.assertEqual(decision.stop_reason, StopReason.QUALITY_THRESHOLD)


class TestAutotunerTierRecommendation(unittest.TestCase):
    """Tests for model tier recommendation based on budget usage."""

    def test_recommends_expensive_when_plenty_budget(self):
        """Should recommend expensive tier when < 30% budget used."""
        config = AutotuneConfig(max_cost_dollars=1.0)
        tuner = Autotuner(config)
        tuner.start()

        # Use ~10% of budget
        tuner.record_round(0, tokens=333, messages=5, support_scores=[0.5])
        # 333 tokens * $0.003/1K = $0.001 = 0.1% of $1

        decision = tuner.should_continue()
        self.assertEqual(decision.recommended_tier, CostTier.EXPENSIVE)

    def test_recommends_standard_when_normal_usage(self):
        """Should recommend standard tier when 30-60% budget used."""
        config = AutotuneConfig(max_cost_dollars=0.01)  # Low budget for easier testing
        tuner = Autotuner(config)
        tuner.start()

        # Use ~45% of budget: $0.0045
        # At $0.003/1K, need 1500 tokens
        tuner.record_round(0, tokens=1500, messages=5, support_scores=[0.5])

        decision = tuner.should_continue()
        self.assertEqual(decision.recommended_tier, CostTier.STANDARD)

    def test_recommends_cheap_when_budget_pressure(self):
        """Should recommend cheap tier when 60-85% budget used."""
        config = AutotuneConfig(max_cost_dollars=0.01)
        tuner = Autotuner(config)
        tuner.start()

        # Use ~75% of budget: $0.0075
        # At $0.003/1K, need 2500 tokens
        tuner.record_round(0, tokens=2500, messages=5, support_scores=[0.5])

        decision = tuner.should_continue()
        self.assertEqual(decision.recommended_tier, CostTier.CHEAP)

    def test_recommends_free_when_critical_budget(self):
        """Should recommend free tier when > 85% budget used."""
        config = AutotuneConfig(max_cost_dollars=0.01)
        tuner = Autotuner(config)
        tuner.start()

        # Use ~90% of budget: $0.009
        # At $0.003/1K, need 3000 tokens
        tuner.record_round(0, tokens=3000, messages=5, support_scores=[0.5])

        decision = tuner.should_continue()
        self.assertEqual(decision.recommended_tier, CostTier.FREE)


class TestAutotunerCostEstimation(unittest.TestCase):
    """Tests for accurate cost estimation."""

    def test_cost_estimation_standard_tier(self):
        """Should estimate cost correctly for standard tier."""
        config = AutotuneConfig()  # Default tier is STANDARD
        tuner = Autotuner(config)
        tuner.start()

        # 10,000 tokens at $0.003/1K = $0.03
        tuner.record_round(0, tokens=10000, messages=5, support_scores=[0.5])

        self.assertAlmostEqual(tuner.metrics.estimated_cost, 0.03, places=5)

    def test_cost_estimation_accumulates(self):
        """Cost should accumulate across rounds."""
        config = AutotuneConfig()
        tuner = Autotuner(config)
        tuner.start()

        tuner.record_round(0, tokens=5000, messages=5, support_scores=[0.5])  # $0.015
        tuner.record_round(1, tokens=5000, messages=5, support_scores=[0.5])  # $0.015

        self.assertAlmostEqual(tuner.metrics.estimated_cost, 0.03, places=5)


class TestAutotunerBudgetRemaining(unittest.TestCase):
    """Tests for get_budget_remaining calculation."""

    def test_budget_remaining_initial(self):
        """Should report full budget remaining initially."""
        config = AutotuneConfig(
            max_cost_dollars=1.0,
            max_tokens=100000,
            max_rounds=5,
        )
        tuner = Autotuner(config)
        tuner.start()

        remaining = tuner.get_budget_remaining()
        self.assertEqual(remaining["cost_remaining"], 1.0)
        self.assertEqual(remaining["tokens_remaining"], 100000)
        self.assertEqual(remaining["rounds_remaining"], 5)
        self.assertEqual(remaining["budget_used_percent"], 0.0)

    def test_budget_remaining_after_usage(self):
        """Should correctly calculate remaining after usage."""
        config = AutotuneConfig(
            max_cost_dollars=1.0,
            max_tokens=100000,
            max_rounds=5,
        )
        tuner = Autotuner(config)
        tuner.start()

        # Use 10,000 tokens = $0.03
        tuner.record_round(0, tokens=10000, messages=5, support_scores=[0.5])

        remaining = tuner.get_budget_remaining()
        self.assertAlmostEqual(remaining["cost_remaining"], 0.97, places=5)
        self.assertEqual(remaining["tokens_remaining"], 90000)
        self.assertEqual(remaining["rounds_remaining"], 4)
        self.assertAlmostEqual(remaining["budget_used_percent"], 3.0, places=3)


class TestAutotunerSuggestRounds(unittest.TestCase):
    """Tests for suggest_rounds optimization."""

    def test_suggest_rounds_no_history(self):
        """Without round history, should use default estimate."""
        config = AutotuneConfig(max_tokens=100000, max_rounds=5)
        tuner = Autotuner(config)
        tuner.start()

        # No rounds completed yet, uses default 5000 tokens/round
        # 100000 / 5000 = 20 rounds affordable, capped at min(20, 5, 3) = 3
        suggested = tuner.suggest_rounds()
        self.assertEqual(suggested, 3)

    def test_suggest_rounds_with_history(self):
        """With history, should estimate from actual usage."""
        config = AutotuneConfig(max_tokens=30000, max_rounds=10)
        tuner = Autotuner(config)
        tuner.start()

        # Used 10000 tokens in 1 round = 10000 tokens/round
        tuner.record_round(0, tokens=10000, messages=5, support_scores=[0.5])

        # Remaining: 20000 tokens / 10000 per round = 2 rounds affordable
        suggested = tuner.suggest_rounds()
        self.assertEqual(suggested, 2)

    def test_suggest_rounds_caps_at_three(self):
        """Should cap suggestion at 3 rounds maximum."""
        config = AutotuneConfig(max_tokens=1000000, max_rounds=100)
        tuner = Autotuner(config)
        tuner.start()

        # Even with massive budget, cap at 3
        suggested = tuner.suggest_rounds()
        self.assertEqual(suggested, 3)

    def test_suggest_rounds_respects_max_remaining(self):
        """Should respect max_rounds limit."""
        config = AutotuneConfig(max_tokens=1000000, max_rounds=2)
        tuner = Autotuner(config)
        tuner.start()

        # Only 2 rounds allowed total
        suggested = tuner.suggest_rounds()
        self.assertEqual(suggested, 2)


class TestAutotunerLifecycle(unittest.TestCase):
    """Tests for start/end lifecycle."""

    def test_start_sets_timestamp(self):
        """start() should record timestamp."""
        tuner = Autotuner()
        self.assertIsNone(tuner._start_time)

        tuner.start()
        self.assertIsNotNone(tuner._start_time)
        self.assertIsNotNone(tuner.metrics.started_at)

    def test_end_records_duration(self):
        """end() should record duration."""
        tuner = Autotuner()
        tuner.start()

        # Small delay
        tuner.end()

        self.assertIsNotNone(tuner.metrics.ended_at)
        self.assertGreater(tuner.metrics.duration_seconds, 0)


class TestAutotunerVerificationDensity(unittest.TestCase):
    """Tests for verification density tracking."""

    def test_verification_density_calculated(self):
        """Should calculate verification density from claims."""
        tuner = Autotuner()
        tuner.start()

        tuner.record_round(
            0,
            tokens=1000,
            messages=5,
            support_scores=[0.8],
            verified_claims=7,
            total_claims=10,
        )

        self.assertAlmostEqual(tuner.metrics.verification_density, 0.7, places=5)
        self.assertEqual(tuner.metrics.claims_made, 10)

    def test_verification_density_zero_claims(self):
        """Should handle zero total claims gracefully."""
        tuner = Autotuner()
        tuner.start()

        tuner.record_round(
            0,
            tokens=1000,
            messages=5,
            support_scores=[0.8],
            verified_claims=0,
            total_claims=0,
        )

        # Should not crash, density stays at 0
        self.assertEqual(tuner.metrics.verification_density, 0.0)


class TestAutotuneDecision(unittest.TestCase):
    """Tests for AutotuneDecision dataclass."""

    def test_decision_defaults(self):
        """Decision should have sensible defaults."""
        decision = AutotuneDecision(should_continue=True)
        self.assertTrue(decision.should_continue)
        self.assertIsNone(decision.stop_reason)
        self.assertEqual(decision.recommended_tier, CostTier.STANDARD)
        self.assertEqual(decision.suggested_action, "")

    def test_decision_with_stop_reason(self):
        """Decision can include stop reason."""
        decision = AutotuneDecision(
            should_continue=False,
            stop_reason=StopReason.MAX_COST,
            suggested_action="Increase budget",
        )
        self.assertFalse(decision.should_continue)
        self.assertEqual(decision.stop_reason, StopReason.MAX_COST)
        self.assertEqual(decision.suggested_action, "Increase budget")


class TestAutotunedDebateRunner(unittest.TestCase):
    """Tests for AutotunedDebateRunner integration wrapper."""

    def test_runner_initializes_tuner(self):
        """Runner should initialize tuner with config."""
        mock_arena = MagicMock()
        config = AutotuneConfig(max_rounds=10)

        runner = AutotunedDebateRunner(mock_arena, config)

        self.assertEqual(runner.tuner.config.max_rounds, 10)
        self.assertIs(runner.arena, mock_arena)

    def test_runner_default_config(self):
        """Runner should use default config if none provided."""
        mock_arena = MagicMock()
        runner = AutotunedDebateRunner(mock_arena)

        self.assertEqual(runner.tuner.config.max_rounds, 5)  # Default


class TestBudgetEnforcementEdgeCases(unittest.TestCase):
    """Edge case tests for budget enforcement - CRITICAL."""

    def test_exact_budget_limit(self):
        """Should stop when exactly at budget limit."""
        config = AutotuneConfig(max_cost_dollars=0.003)  # Exactly 1K tokens at STANDARD
        tuner = Autotuner(config)
        tuner.start()

        tuner.record_round(0, tokens=1000, messages=5, support_scores=[0.5])

        decision = tuner.should_continue()
        self.assertFalse(decision.should_continue)
        self.assertEqual(decision.stop_reason, StopReason.MAX_COST)

    def test_exact_token_limit(self):
        """Should stop when exactly at token limit."""
        config = AutotuneConfig(max_tokens=5000)
        tuner = Autotuner(config)
        tuner.start()

        tuner.record_round(0, tokens=5000, messages=5, support_scores=[0.5])

        decision = tuner.should_continue()
        self.assertFalse(decision.should_continue)
        self.assertEqual(decision.stop_reason, StopReason.MAX_TOKENS)

    def test_multiple_limits_hit_simultaneously(self):
        """When multiple limits hit, first check wins (rounds -> cost -> tokens)."""
        config = AutotuneConfig(
            max_rounds=1,
            max_cost_dollars=0.003,
            max_tokens=1000,
        )
        tuner = Autotuner(config)
        tuner.start()

        tuner.record_round(0, tokens=1000, messages=5, support_scores=[0.5])

        decision = tuner.should_continue()
        # MAX_ROUNDS is checked first
        self.assertEqual(decision.stop_reason, StopReason.MAX_ROUNDS)

    def test_zero_budget_stops_immediately(self):
        """Zero budget should stop on first usage."""
        config = AutotuneConfig(max_cost_dollars=0.0)
        tuner = Autotuner(config)
        tuner.start()

        tuner.record_round(0, tokens=1, messages=1, support_scores=[0.5])

        decision = tuner.should_continue()
        self.assertFalse(decision.should_continue)
        # Will hit cost limit since 0 >= 0
        self.assertEqual(decision.stop_reason, StopReason.MAX_COST)


class TestConsensusRecording(unittest.TestCase):
    """Tests for consensus metric recording."""

    def test_record_consensus_updates_confidence(self):
        """record_consensus should update confidence metric."""
        tuner = Autotuner()
        tuner.start()

        tuner.record_consensus(confidence=0.92, reached=True)

        self.assertEqual(tuner.metrics.consensus_confidence, 0.92)

    def test_record_consensus_multiple_times(self):
        """Last consensus confidence should be recorded."""
        tuner = Autotuner()
        tuner.start()

        tuner.record_consensus(confidence=0.5, reached=False)
        tuner.record_consensus(confidence=0.75, reached=False)
        tuner.record_consensus(confidence=0.88, reached=True)

        self.assertEqual(tuner.metrics.consensus_confidence, 0.88)


if __name__ == "__main__":
    unittest.main()
