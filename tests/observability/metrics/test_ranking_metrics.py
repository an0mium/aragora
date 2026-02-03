"""Tests for observability/metrics/ranking.py — ranking and agent selection metrics."""

from unittest.mock import patch

import pytest

from aragora.observability.metrics import ranking as mod
from aragora.observability.metrics.ranking import (
    init_ranking_metrics,
    record_analytics_selection_recommendation,
    record_budget_filtering_event,
    record_calibration_adjustment,
    record_calibration_cost_calculation,
    record_echo_chamber_detection,
    record_learning_bonus,
    record_novelty_penalty,
    record_novelty_score_calculation,
    record_outcome_complexity_adjustment,
    record_performance_routing_decision,
    record_performance_routing_latency,
    record_relationship_bias_adjustment,
    record_rlm_selection_recommendation,
    record_selection_feedback_adjustment,
    record_voting_accuracy_update,
    track_performance_routing,
)


@pytest.fixture(autouse=True)
def _reset_module():
    mod._initialized = False
    yield
    mod._initialized = False


@pytest.fixture()
def _init_noop():
    with patch("aragora.observability.metrics.ranking.get_metrics_enabled", return_value=False):
        init_ranking_metrics()


class TestInitialization:
    def test_init_noop(self, _init_noop):
        assert mod._initialized is True
        assert mod.CALIBRATION_ADJUSTMENTS is not None

    def test_init_idempotent(self, _init_noop):
        first = mod.CALIBRATION_ADJUSTMENTS
        init_ranking_metrics()
        assert mod.CALIBRATION_ADJUSTMENTS is first


class TestCoreELOMetrics:
    def test_record_calibration_adjustment(self, _init_noop):
        record_calibration_adjustment("claude")

    def test_record_learning_bonus(self, _init_noop):
        record_learning_bonus("gpt-4", "improvement")

    def test_record_voting_accuracy_update(self, _init_noop):
        record_voting_accuracy_update("correct")
        record_voting_accuracy_update("incorrect")

    def test_record_selection_feedback_adjustment(self, _init_noop):
        record_selection_feedback_adjustment("claude", "increase")
        record_selection_feedback_adjustment("gpt-4", "decrease")


class TestPerformanceRoutingMetrics:
    def test_record_routing_decision(self, _init_noop):
        record_performance_routing_decision("critique", "claude")

    def test_record_routing_latency(self, _init_noop):
        record_performance_routing_latency(0.005)

    def test_track_performance_routing(self, _init_noop):
        with track_performance_routing("proposal") as ctx:
            ctx["selected_agent"] = "gpt-4"

    def test_track_performance_routing_default(self, _init_noop):
        with track_performance_routing("vote") as ctx:
            pass  # selected_agent defaults to "unknown"


class TestNoveltyDiversityMetrics:
    def test_record_novelty_score(self, _init_noop):
        record_novelty_score_calculation("claude")

    def test_record_novelty_penalty(self, _init_noop):
        record_novelty_penalty("gpt-4")

    def test_record_echo_chamber_detection(self, _init_noop):
        record_echo_chamber_detection("high")
        record_echo_chamber_detection("low")

    def test_record_relationship_bias_adjustment(self, _init_noop):
        record_relationship_bias_adjustment("claude", "decrease")


class TestRLMCostMetrics:
    def test_record_rlm_selection(self, _init_noop):
        record_rlm_selection_recommendation("gemini")

    def test_record_calibration_cost(self, _init_noop):
        record_calibration_cost_calculation("claude", "high")

    def test_record_budget_filtering(self, _init_noop):
        record_budget_filtering_event("filtered")
        record_budget_filtering_event("passed")
        record_budget_filtering_event("bypassed")


class TestAnalyticsSelectionMetrics:
    def test_record_outcome_complexity(self, _init_noop):
        record_outcome_complexity_adjustment("increase")
        record_outcome_complexity_adjustment("decrease")

    def test_record_analytics_recommendation(self, _init_noop):
        record_analytics_selection_recommendation("add_specialist")
        record_analytics_selection_recommendation("reduce_team_size")
