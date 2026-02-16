"""Tests for the SelfCorrectionEngine (self-correcting improvement loop)."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta

from aragora.nomic.self_correction import (
    CorrectionReport,
    SelfCorrectionConfig,
    SelfCorrectionEngine,
    StrategyRecommendation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> SelfCorrectionEngine:
    """Create a default SelfCorrectionEngine."""
    return SelfCorrectionEngine()


@pytest.fixture
def engine_low_threshold() -> SelfCorrectionEngine:
    """Engine with lower thresholds for easier pattern detection."""
    return SelfCorrectionEngine(
        config=SelfCorrectionConfig(
            min_cycles_for_pattern=2,
            failure_repeat_threshold=2,
        )
    )


def _make_outcome(
    track: str,
    success: bool,
    agent: str | None = None,
    goal_type: str | None = None,
    timestamp: str | None = None,
    description: str | None = None,
) -> dict:
    """Helper to build an outcome dict."""
    outcome: dict = {"track": track, "success": success}
    if agent is not None:
        outcome["agent"] = agent
    if goal_type is not None:
        outcome["goal_type"] = goal_type
    if timestamp is not None:
        outcome["timestamp"] = timestamp
    if description is not None:
        outcome["description"] = description
    return outcome


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# 1. analyze_patterns identifies track success rates
# ---------------------------------------------------------------------------


class TestAnalyzePatternsTrackRates:
    """analyze_patterns should correctly compute track success rates."""

    def test_basic_track_success_rates(self, engine: SelfCorrectionEngine):
        outcomes = [
            _make_outcome("qa", True),
            _make_outcome("qa", True),
            _make_outcome("qa", False),
            _make_outcome("core", False),
            _make_outcome("core", False),
        ]
        report = engine.analyze_patterns(outcomes)
        # qa: 2/3 successes
        assert abs(report.track_success_rates["qa"] - 2 / 3) < 0.01
        # core: 0/2 successes
        assert report.track_success_rates["core"] == 0.0

    def test_overall_success_rate(self, engine: SelfCorrectionEngine):
        outcomes = [
            _make_outcome("sme", True),
            _make_outcome("sme", False),
            _make_outcome("developer", True),
            _make_outcome("developer", True),
        ]
        report = engine.analyze_patterns(outcomes)
        # 3/4 total successes
        assert abs(report.overall_success_rate - 0.75) < 0.01
        assert report.total_cycles == 4

    def test_single_track_all_successes(self, engine: SelfCorrectionEngine):
        outcomes = [_make_outcome("security", True) for _ in range(5)]
        report = engine.analyze_patterns(outcomes)
        assert report.track_success_rates["security"] == 1.0


# ---------------------------------------------------------------------------
# 2. analyze_patterns detects consecutive failure streaks
# ---------------------------------------------------------------------------


class TestAnalyzePatternsStreaks:
    """analyze_patterns should detect consecutive success/failure streaks."""

    def test_consecutive_failure_streak(self, engine: SelfCorrectionEngine):
        outcomes = [
            _make_outcome("core", True),
            _make_outcome("core", False),
            _make_outcome("core", False),
            _make_outcome("core", False),
        ]
        report = engine.analyze_patterns(outcomes)
        # 3 consecutive failures at the end -> streak = -3
        assert report.track_streaks["core"] == -3

    def test_consecutive_success_streak(self, engine: SelfCorrectionEngine):
        outcomes = [
            _make_outcome("qa", False),
            _make_outcome("qa", True),
            _make_outcome("qa", True),
            _make_outcome("qa", True),
        ]
        report = engine.analyze_patterns(outcomes)
        # 3 consecutive successes at the end -> streak = +3
        assert report.track_streaks["qa"] == 3

    def test_mixed_tracks_independent_streaks(self, engine: SelfCorrectionEngine):
        outcomes = [
            _make_outcome("core", False),
            _make_outcome("qa", True),
            _make_outcome("core", False),
            _make_outcome("qa", True),
            _make_outcome("qa", True),
        ]
        report = engine.analyze_patterns(outcomes)
        assert report.track_streaks["core"] == -2
        assert report.track_streaks["qa"] == 3

    def test_failing_patterns_detected_on_streak(self, engine: SelfCorrectionEngine):
        outcomes = [
            _make_outcome("core", False),
            _make_outcome("core", False),
            _make_outcome("core", False),
        ]
        report = engine.analyze_patterns(outcomes)
        assert any("core" in p and "consecutive" in p for p in report.failing_patterns)


# ---------------------------------------------------------------------------
# 3. compute_priority_adjustments boosts successful tracks
# ---------------------------------------------------------------------------


class TestPriorityAdjustmentsBoost:
    """compute_priority_adjustments should boost tracks with success momentum."""

    def test_successful_track_gets_boost(self, engine: SelfCorrectionEngine):
        outcomes = [
            _make_outcome("qa", True),
            _make_outcome("qa", True),
            _make_outcome("qa", True),
        ]
        report = engine.analyze_patterns(outcomes)
        adjustments = engine.compute_priority_adjustments(report)
        # Positive streak of 3 -> 1.0 + (0.1 * 3) + 0.05 = 1.35
        assert adjustments["qa"] > 1.0

    def test_multiple_tracks_differentiated(self, engine: SelfCorrectionEngine):
        outcomes = [
            _make_outcome("qa", True),
            _make_outcome("qa", True),
            _make_outcome("qa", True),
            _make_outcome("core", True),
            _make_outcome("core", False),
            _make_outcome("core", True),
        ]
        report = engine.analyze_patterns(outcomes)
        adjustments = engine.compute_priority_adjustments(report)
        # qa has perfect streak, core has mixed
        assert adjustments["qa"] > adjustments["core"]


# ---------------------------------------------------------------------------
# 4. compute_priority_adjustments deprioritizes failing tracks
# ---------------------------------------------------------------------------


class TestPriorityAdjustmentsDeprioritize:
    """compute_priority_adjustments should deprioritize failing tracks."""

    def test_failing_track_gets_penalty(self, engine: SelfCorrectionEngine):
        outcomes = [
            _make_outcome("core", True),
            _make_outcome("core", False),
            _make_outcome("core", False),
            _make_outcome("core", False),
        ]
        report = engine.analyze_patterns(outcomes)
        adjustments = engine.compute_priority_adjustments(report)
        # Negative streak of -3 (>= threshold of 2) -> penalty applied
        assert adjustments["core"] < 1.0

    def test_long_failure_streak_stronger_penalty(self, engine: SelfCorrectionEngine):
        outcomes = [_make_outcome("core", False) for _ in range(5)]
        report = engine.analyze_patterns(outcomes)
        adjustments = engine.compute_priority_adjustments(report)
        # 5 consecutive failures + 0% rate -> significant reduction
        assert adjustments["core"] < 0.5

    def test_adjustment_clamped_to_minimum(self, engine: SelfCorrectionEngine):
        """Even extreme failures should not produce adjustment below 0.1."""
        outcomes = [_make_outcome("core", False) for _ in range(10)]
        report = engine.analyze_patterns(outcomes)
        adjustments = engine.compute_priority_adjustments(report)
        assert adjustments["core"] >= 0.1


# ---------------------------------------------------------------------------
# 5. recommend_strategy_change suggests agent rotation after repeated failures
# ---------------------------------------------------------------------------


class TestStrategyChangeAgentRotation:
    """recommend_strategy_change should suggest agent rotation on failure."""

    def test_low_correlation_agent_flagged(self, engine: SelfCorrectionEngine):
        outcomes = [
            _make_outcome("core", False, agent="deepseek"),
            _make_outcome("core", False, agent="deepseek"),
            _make_outcome("core", False, agent="deepseek"),
            _make_outcome("qa", True, agent="claude"),
        ]
        report = engine.analyze_patterns(outcomes)
        recommendations = engine.recommend_strategy_change(report)
        rotate_recs = [r for r in recommendations if r.action_type == "rotate_agent"]
        assert len(rotate_recs) >= 1
        assert any("deepseek" in r.recommendation for r in rotate_recs)

    def test_agent_rotation_includes_confidence(self, engine: SelfCorrectionEngine):
        outcomes = [
            _make_outcome("qa", False, agent="grok"),
            _make_outcome("qa", False, agent="grok"),
            _make_outcome("qa", False, agent="grok"),
            _make_outcome("qa", True, agent="claude"),
        ]
        report = engine.analyze_patterns(outcomes)
        recommendations = engine.recommend_strategy_change(report)
        agent_recs = [r for r in recommendations if r.action_type == "rotate_agent"]
        for rec in agent_recs:
            assert 0.0 < rec.confidence <= 1.0


# ---------------------------------------------------------------------------
# 6. recommend_strategy_change suggests scope reduction after failures
# ---------------------------------------------------------------------------


class TestStrategyChangeScopeReduction:
    """recommend_strategy_change should suggest scope reduction on streak failures."""

    def test_consecutive_failures_trigger_scope_decrease(
        self, engine: SelfCorrectionEngine
    ):
        outcomes = [
            _make_outcome("core", True),
            _make_outcome("core", False),
            _make_outcome("core", False),
            _make_outcome("core", False),
        ]
        report = engine.analyze_patterns(outcomes)
        recommendations = engine.recommend_strategy_change(report)
        scope_recs = [r for r in recommendations if r.action_type == "decrease_scope"]
        assert len(scope_recs) >= 1
        assert any("core" in r.track for r in scope_recs)
        assert any("incremental" in r.recommendation.lower() for r in scope_recs)

    def test_longer_streak_higher_confidence(self, engine: SelfCorrectionEngine):
        short_outcomes = [
            _make_outcome("core", False),
            _make_outcome("core", False),
            _make_outcome("core", False),
        ]
        long_outcomes = [_make_outcome("core", False) for _ in range(6)]

        report_short = engine.analyze_patterns(short_outcomes)
        report_long = engine.analyze_patterns(long_outcomes)

        recs_short = engine.recommend_strategy_change(report_short)
        recs_long = engine.recommend_strategy_change(report_long)

        scope_short = [r for r in recs_short if r.action_type == "decrease_scope"]
        scope_long = [r for r in recs_long if r.action_type == "decrease_scope"]

        assert scope_short and scope_long
        assert scope_long[0].confidence >= scope_short[0].confidence

    def test_deprioritize_recommendation_on_extreme_failure(
        self, engine: SelfCorrectionEngine
    ):
        """Tracks with very low rate and long streak should get deprioritize rec."""
        outcomes = [_make_outcome("core", False) for _ in range(5)]
        report = engine.analyze_patterns(outcomes)
        recommendations = engine.recommend_strategy_change(report)
        depri_recs = [r for r in recommendations if r.action_type == "deprioritize"]
        assert len(depri_recs) >= 1
        assert any("core" in r.track for r in depri_recs)


# ---------------------------------------------------------------------------
# 7. Ignores patterns older than max_pattern_age_days
# ---------------------------------------------------------------------------


class TestMaxPatternAge:
    """Outcomes older than max_pattern_age_days should be excluded."""

    def test_old_outcomes_filtered_out(self):
        engine = SelfCorrectionEngine(
            config=SelfCorrectionConfig(max_pattern_age_days=7)
        )
        old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        recent_ts = datetime.now(timezone.utc).isoformat()

        outcomes = [
            _make_outcome("qa", False, timestamp=old_ts),
            _make_outcome("qa", False, timestamp=old_ts),
            _make_outcome("qa", True, timestamp=recent_ts),
        ]
        report = engine.analyze_patterns(outcomes)
        # Only the recent outcome should be counted
        assert report.total_cycles == 1
        assert report.track_success_rates["qa"] == 1.0

    def test_outcomes_without_timestamp_are_included(self):
        engine = SelfCorrectionEngine(
            config=SelfCorrectionConfig(max_pattern_age_days=7)
        )
        outcomes = [
            _make_outcome("qa", True),  # No timestamp -> included
            _make_outcome("qa", True),
        ]
        report = engine.analyze_patterns(outcomes)
        assert report.total_cycles == 2

    def test_invalid_timestamp_treated_as_recent(self):
        engine = SelfCorrectionEngine(
            config=SelfCorrectionConfig(max_pattern_age_days=7)
        )
        outcomes = [
            _make_outcome("qa", True, timestamp="not-a-date"),
        ]
        report = engine.analyze_patterns(outcomes)
        assert report.total_cycles == 1


# ---------------------------------------------------------------------------
# 8. Returns neutral adjustments when insufficient data
# ---------------------------------------------------------------------------


class TestInsufficientData:
    """Engine should return neutral results when < min_cycles_for_pattern."""

    def test_neutral_adjustments_with_few_cycles(self, engine: SelfCorrectionEngine):
        """Default min_cycles_for_pattern=3; with 2 outcomes, all adjustments=1.0."""
        outcomes = [
            _make_outcome("qa", True),
            _make_outcome("core", False),
        ]
        report = engine.analyze_patterns(outcomes)
        adjustments = engine.compute_priority_adjustments(report)
        # With only 2 cycles (< 3), all adjustments should be neutral
        assert adjustments.get("qa") == 1.0
        assert adjustments.get("core") == 1.0

    def test_no_recommendations_with_few_cycles(self, engine: SelfCorrectionEngine):
        outcomes = [
            _make_outcome("core", False),
            _make_outcome("core", False),
        ]
        report = engine.analyze_patterns(outcomes)
        recommendations = engine.recommend_strategy_change(report)
        assert recommendations == []

    def test_exact_threshold_produces_results(self):
        """When cycles == min_cycles_for_pattern, results should be non-neutral."""
        engine = SelfCorrectionEngine(
            config=SelfCorrectionConfig(min_cycles_for_pattern=3)
        )
        outcomes = [
            _make_outcome("qa", True),
            _make_outcome("qa", True),
            _make_outcome("qa", True),
        ]
        report = engine.analyze_patterns(outcomes)
        adjustments = engine.compute_priority_adjustments(report)
        # With exactly 3 cycles (>= 3), adjustments should be computed
        assert adjustments["qa"] > 1.0


# ---------------------------------------------------------------------------
# 9. Works with empty outcome list
# ---------------------------------------------------------------------------


class TestEmptyOutcomes:
    """Engine should handle empty input gracefully."""

    def test_analyze_empty_list(self, engine: SelfCorrectionEngine):
        report = engine.analyze_patterns([])
        assert report.total_cycles == 0
        assert report.overall_success_rate == 0.0
        assert report.track_success_rates == {}
        assert report.track_streaks == {}
        assert report.agent_correlations == {}
        assert report.failing_patterns == []

    def test_compute_adjustments_from_empty_report(self, engine: SelfCorrectionEngine):
        report = engine.analyze_patterns([])
        adjustments = engine.compute_priority_adjustments(report)
        assert adjustments == {}

    def test_recommend_from_empty_report(self, engine: SelfCorrectionEngine):
        report = engine.analyze_patterns([])
        recommendations = engine.recommend_strategy_change(report)
        assert recommendations == []


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Additional edge case tests."""

    def test_agent_correlations_computed(self, engine: SelfCorrectionEngine):
        outcomes = [
            _make_outcome("qa", True, agent="claude"),
            _make_outcome("qa", True, agent="claude"),
            _make_outcome("qa", False, agent="deepseek"),
            _make_outcome("core", False, agent="deepseek"),
        ]
        report = engine.analyze_patterns(outcomes)
        assert report.agent_correlations["claude"] == 1.0
        assert report.agent_correlations["deepseek"] == 0.0

    def test_goal_type_failing_patterns(self, engine: SelfCorrectionEngine):
        outcomes = [
            _make_outcome("qa", False, goal_type="refactor"),
            _make_outcome("qa", False, goal_type="refactor"),
            _make_outcome("qa", True, goal_type="bugfix"),
            _make_outcome("core", True, goal_type="bugfix"),
        ]
        report = engine.analyze_patterns(outcomes)
        assert any("refactor" in p for p in report.failing_patterns)

    def test_config_customization(self):
        config = SelfCorrectionConfig(
            min_cycles_for_pattern=5,
            failure_repeat_threshold=3,
            success_momentum_bonus=0.2,
            failure_penalty=0.25,
            max_pattern_age_days=14,
        )
        engine = SelfCorrectionEngine(config=config)
        assert engine.config.min_cycles_for_pattern == 5
        assert engine.config.failure_penalty == 0.25

    def test_outcomes_without_agent_skip_agent_correlation(
        self, engine: SelfCorrectionEngine
    ):
        outcomes = [
            _make_outcome("qa", True),
            _make_outcome("qa", False),
            _make_outcome("core", True),
        ]
        report = engine.analyze_patterns(outcomes)
        # No agent field -> empty correlations
        assert report.agent_correlations == {}

    def test_strategy_recommendation_dataclass_fields(self):
        rec = StrategyRecommendation(
            track="core",
            recommendation="Do something",
            reason="Because reasons",
            confidence=0.8,
            action_type="decrease_scope",
        )
        assert rec.track == "core"
        assert rec.action_type == "decrease_scope"
        assert rec.confidence == 0.8

    def test_correction_report_dataclass_fields(self):
        report = CorrectionReport(
            total_cycles=10,
            overall_success_rate=0.7,
            track_success_rates={"qa": 0.8},
            track_streaks={"qa": 3},
            agent_correlations={"claude": 0.9},
            failing_patterns=["test pattern"],
        )
        assert report.total_cycles == 10
        assert report.overall_success_rate == 0.7
        assert report.failing_patterns == ["test pattern"]
