"""Tests for PerformanceEloIntegrator."""

from __future__ import annotations

import pytest
from collections import defaultdict
from unittest.mock import Mock, MagicMock

from aragora.ranking.performance_integrator import (
    PerformanceEloIntegrator,
    PerformanceScore,
    create_performance_integrator,
)
from aragora.agents.performance_monitor import AgentStats, AgentMetric


class TestPerformanceScore:
    """Test PerformanceScore dataclass."""

    def test_default_values(self) -> None:
        """Default scores are neutral (0.5)."""
        score = PerformanceScore(agent_name="test")
        assert score.response_quality_score == 0.5
        assert score.latency_score == 0.5
        assert score.consistency_score == 0.5
        assert score.participation_score == 0.5
        assert score.composite_score == 0.5

    def test_to_dict(self) -> None:
        """to_dict returns proper serialization."""
        score = PerformanceScore(
            agent_name="test",
            response_quality_score=0.8,
            latency_score=0.6,
            consistency_score=0.7,
            participation_score=0.9,
            composite_score=0.75,
        )
        d = score.to_dict()
        assert d["agent_name"] == "test"
        assert d["response_quality_score"] == 0.8
        assert d["composite_score"] == 0.75


class TestPerformanceEloIntegrator:
    """Test PerformanceEloIntegrator core functionality."""

    def test_init_without_dependencies(self) -> None:
        """Integrator can be created without dependencies."""
        integrator = PerformanceEloIntegrator()
        assert integrator.performance_monitor is None
        assert integrator.elo_system is None

    def test_weights_normalized(self) -> None:
        """Weights are normalized to sum to 1.0."""
        integrator = PerformanceEloIntegrator(
            response_quality_weight=2.0,
            latency_weight=1.0,
            consistency_weight=1.0,
            participation_weight=1.0,
        )
        total = (
            integrator.response_quality_weight
            + integrator.latency_weight
            + integrator.consistency_weight
            + integrator.participation_weight
        )
        assert abs(total - 1.0) < 0.001

    def test_compute_score_no_monitor(self) -> None:
        """Returns neutral score when no monitor is configured."""
        integrator = PerformanceEloIntegrator()
        score = integrator.compute_performance_score("test")
        assert score.composite_score == 0.5

    def test_compute_score_insufficient_data(self) -> None:
        """Returns neutral score when agent has few calls."""
        monitor = Mock()
        stats = AgentStats()
        stats.total_calls = 2  # Below min_calls_for_adjustment
        monitor.agent_stats = {"test": stats}

        integrator = PerformanceEloIntegrator(performance_monitor=monitor)
        score = integrator.compute_performance_score("test")
        assert score.composite_score == 0.5

    def test_compute_score_high_success_rate(self) -> None:
        """High success rate results in high quality score."""
        monitor = Mock()
        stats = AgentStats()
        stats.total_calls = 10
        stats.successful_calls = 9
        stats.failed_calls = 1
        stats.total_duration_ms = 10000
        stats.avg_duration_ms = 1000
        stats.min_duration_ms = 500
        stats.max_duration_ms = 1500
        monitor.agent_stats = {"test": stats}

        integrator = PerformanceEloIntegrator(performance_monitor=monitor)
        score = integrator.compute_performance_score("test")

        # 90% success rate
        assert score.response_quality_score == 0.9
        assert score.composite_score > 0.5

    def test_compute_score_low_success_rate(self) -> None:
        """Low success rate results in low quality score."""
        monitor = Mock()
        stats = AgentStats()
        stats.total_calls = 10
        stats.successful_calls = 3
        stats.failed_calls = 7
        stats.total_duration_ms = 500000
        stats.avg_duration_ms = 50000  # Slow
        stats.min_duration_ms = 10000
        stats.max_duration_ms = 100000  # Inconsistent
        monitor.agent_stats = {"test": stats}

        integrator = PerformanceEloIntegrator(performance_monitor=monitor)
        score = integrator.compute_performance_score("test")

        # 30% success rate should be reflected
        assert score.response_quality_score == 0.3
        # With low quality + slow + inconsistent, composite should be low
        assert score.composite_score < 0.5

    def test_compute_score_fast_agent(self) -> None:
        """Fast agent gets high latency score."""
        monitor = Mock()
        stats = AgentStats()
        stats.total_calls = 10
        stats.successful_calls = 10
        stats.failed_calls = 0
        stats.total_duration_ms = 5000
        stats.avg_duration_ms = 500  # Very fast
        stats.min_duration_ms = 400
        stats.max_duration_ms = 600
        monitor.agent_stats = {"test": stats}

        integrator = PerformanceEloIntegrator(performance_monitor=monitor)
        score = integrator.compute_performance_score("test")

        # Fast response (500ms / 120000ms max = very high score)
        assert score.latency_score > 0.9

    def test_compute_score_slow_agent(self) -> None:
        """Slow agent gets low latency score."""
        monitor = Mock()
        stats = AgentStats()
        stats.total_calls = 10
        stats.successful_calls = 10
        stats.failed_calls = 0
        stats.total_duration_ms = 1000000
        stats.avg_duration_ms = 100000  # 100 seconds
        stats.min_duration_ms = 80000
        stats.max_duration_ms = 120000
        monitor.agent_stats = {"test": stats}

        integrator = PerformanceEloIntegrator(performance_monitor=monitor)
        score = integrator.compute_performance_score("test")

        # Slow response
        assert score.latency_score < 0.2

    def test_compute_score_consistent_agent(self) -> None:
        """Consistent agent (low variance) gets high consistency score."""
        monitor = Mock()
        stats = AgentStats()
        stats.total_calls = 10
        stats.successful_calls = 10
        stats.failed_calls = 0
        stats.total_duration_ms = 10000
        stats.avg_duration_ms = 1000
        stats.min_duration_ms = 950  # Very consistent
        stats.max_duration_ms = 1050
        monitor.agent_stats = {"test": stats}

        integrator = PerformanceEloIntegrator(performance_monitor=monitor)
        score = integrator.compute_performance_score("test")

        # Low variance = high consistency
        assert score.consistency_score > 0.9

    def test_compute_score_inconsistent_agent(self) -> None:
        """Inconsistent agent (high variance) gets low consistency score."""
        monitor = Mock()
        stats = AgentStats()
        stats.total_calls = 10
        stats.successful_calls = 10
        stats.failed_calls = 0
        stats.total_duration_ms = 100000
        stats.avg_duration_ms = 10000
        stats.min_duration_ms = 1000  # Very inconsistent
        stats.max_duration_ms = 60000
        monitor.agent_stats = {"test": stats}

        integrator = PerformanceEloIntegrator(performance_monitor=monitor)
        score = integrator.compute_performance_score("test")

        # High variance = low consistency
        assert score.consistency_score < 0.3

    def test_compute_score_high_participation(self) -> None:
        """High participation (many calls) gets high participation score."""
        monitor = Mock()
        stats = AgentStats()
        stats.total_calls = 50  # Many calls
        stats.successful_calls = 40
        stats.failed_calls = 10
        stats.total_duration_ms = 50000
        stats.avg_duration_ms = 1000
        stats.min_duration_ms = 500
        stats.max_duration_ms = 1500
        monitor.agent_stats = {"test": stats}

        integrator = PerformanceEloIntegrator(performance_monitor=monitor)
        score = integrator.compute_performance_score("test")

        # At saturation point (20+), should be 1.0
        assert score.participation_score == 1.0


class TestKFactorMultipliers:
    """Test K-factor multiplier computation."""

    def test_multipliers_neutral_without_data(self) -> None:
        """Without data, multipliers are near the middle of range."""
        integrator = PerformanceEloIntegrator()
        multipliers = integrator.compute_k_multipliers(["agent1", "agent2"])

        # Neutral score (0.5) should give middle multiplier
        min_k, max_k = integrator.k_factor_range
        expected = max_k - (0.5 * (max_k - min_k))

        assert abs(multipliers["agent1"] - expected) < 0.01
        assert abs(multipliers["agent2"] - expected) < 0.01

    def test_multipliers_high_performer_stable(self) -> None:
        """High performer gets lower K-factor (more stable ELO)."""
        monitor = Mock()
        stats = AgentStats()
        stats.total_calls = 20
        stats.successful_calls = 19
        stats.failed_calls = 1
        stats.total_duration_ms = 10000
        stats.avg_duration_ms = 500
        stats.min_duration_ms = 400
        stats.max_duration_ms = 600
        monitor.agent_stats = {"high_performer": stats}

        integrator = PerformanceEloIntegrator(performance_monitor=monitor)
        multipliers = integrator.compute_k_multipliers(["high_performer"])

        # High score should result in low K-factor
        min_k, max_k = integrator.k_factor_range
        assert multipliers["high_performer"] < (max_k + min_k) / 2

    def test_multipliers_low_performer_volatile(self) -> None:
        """Low performer gets higher K-factor (more volatile ELO)."""
        monitor = Mock()
        stats = AgentStats()
        stats.total_calls = 20
        stats.successful_calls = 5
        stats.failed_calls = 15
        stats.total_duration_ms = 200000
        stats.avg_duration_ms = 100000  # Slow
        stats.min_duration_ms = 10000
        stats.max_duration_ms = 200000  # Inconsistent
        monitor.agent_stats = {"low_performer": stats}

        integrator = PerformanceEloIntegrator(performance_monitor=monitor)
        multipliers = integrator.compute_k_multipliers(["low_performer"])

        # Low score should result in high K-factor
        min_k, max_k = integrator.k_factor_range
        assert multipliers["low_performer"] > (max_k + min_k) / 2


class TestDegradedAgents:
    """Test degraded agent detection."""

    def test_no_degraded_without_monitor(self) -> None:
        """Returns empty list without monitor."""
        integrator = PerformanceEloIntegrator()
        degraded = integrator.get_degraded_agents()
        assert degraded == []

    def test_detects_degraded_agents(self) -> None:
        """Detects agents below threshold."""
        monitor = Mock()

        # Good agent
        good_stats = AgentStats()
        good_stats.total_calls = 10
        good_stats.successful_calls = 9
        good_stats.failed_calls = 1
        good_stats.avg_duration_ms = 1000
        good_stats.min_duration_ms = 500
        good_stats.max_duration_ms = 1500

        # Bad agent
        bad_stats = AgentStats()
        bad_stats.total_calls = 10
        bad_stats.successful_calls = 2
        bad_stats.failed_calls = 8
        bad_stats.avg_duration_ms = 100000
        bad_stats.min_duration_ms = 10000
        bad_stats.max_duration_ms = 200000

        monitor.agent_stats = {
            "good_agent": good_stats,
            "bad_agent": bad_stats,
        }

        integrator = PerformanceEloIntegrator(performance_monitor=monitor)
        degraded = integrator.get_degraded_agents(threshold=0.3)

        assert "bad_agent" in degraded
        assert "good_agent" not in degraded


class TestCalibrationInterface:
    """Test CalibrationTracker interface compatibility."""

    def test_calibration_summary_returns_proxy(self) -> None:
        """get_calibration_summary returns CalibrationTracker-compatible object."""
        monitor = Mock()
        stats = AgentStats()
        stats.total_calls = 10
        stats.successful_calls = 8
        stats.failed_calls = 2
        stats.avg_duration_ms = 1000
        stats.min_duration_ms = 500
        stats.max_duration_ms = 1500
        monitor.agent_stats = {"test": stats}

        integrator = PerformanceEloIntegrator(performance_monitor=monitor)
        summary = integrator.get_calibration_summary("test")

        # Should have CalibrationTracker-like attributes
        assert hasattr(summary, "calibration_score")
        assert hasattr(summary, "total_predictions")
        assert hasattr(summary, "brier_score")

        assert summary.total_predictions == 10
        assert 0 <= summary.calibration_score <= 1
        assert 0 <= summary.brier_score <= 1


class TestPerformanceSummary:
    """Test performance summary generation."""

    def test_summary_without_monitor(self) -> None:
        """Returns message when no monitor."""
        integrator = PerformanceEloIntegrator()
        summary = integrator.get_performance_summary()
        assert "message" in summary

    def test_summary_with_agents(self) -> None:
        """Returns full summary with agents."""
        monitor = Mock()

        stats1 = AgentStats()
        stats1.total_calls = 10
        stats1.successful_calls = 9
        stats1.failed_calls = 1
        stats1.avg_duration_ms = 1000
        stats1.min_duration_ms = 500
        stats1.max_duration_ms = 1500

        stats2 = AgentStats()
        stats2.total_calls = 10
        stats2.successful_calls = 3
        stats2.failed_calls = 7
        stats2.avg_duration_ms = 50000
        stats2.min_duration_ms = 10000
        stats2.max_duration_ms = 100000

        monitor.agent_stats = {
            "good_agent": stats1,
            "bad_agent": stats2,
        }

        integrator = PerformanceEloIntegrator(performance_monitor=monitor)
        summary = integrator.get_performance_summary()

        assert "agents" in summary
        assert "top_performers" in summary
        assert "degraded" in summary
        assert "recommendations" in summary

        assert "good_agent" in summary["agents"]
        assert "bad_agent" in summary["agents"]


class TestCaching:
    """Test score caching."""

    def test_scores_are_cached(self) -> None:
        """Computed scores are cached."""
        monitor = Mock()
        stats = AgentStats()
        stats.total_calls = 10
        stats.successful_calls = 9
        stats.failed_calls = 1
        stats.avg_duration_ms = 1000
        stats.min_duration_ms = 500
        stats.max_duration_ms = 1500
        monitor.agent_stats = {"test": stats}

        integrator = PerformanceEloIntegrator(performance_monitor=monitor)

        # First call computes
        score1 = integrator.compute_performance_score("test")

        # Second call returns cached
        cached = integrator.get_cached_score("test")
        assert cached is not None
        assert cached.composite_score == score1.composite_score

    def test_clear_cache(self) -> None:
        """clear_cache removes all cached scores."""
        monitor = Mock()
        stats = AgentStats()
        stats.total_calls = 10
        stats.successful_calls = 9
        stats.failed_calls = 1
        stats.avg_duration_ms = 1000
        stats.min_duration_ms = 500
        stats.max_duration_ms = 1500
        monitor.agent_stats = {"test": stats}

        integrator = PerformanceEloIntegrator(performance_monitor=monitor)
        integrator.compute_performance_score("test")

        assert integrator.get_cached_score("test") is not None

        integrator.clear_cache()

        assert integrator.get_cached_score("test") is None


class TestCreateHelper:
    """Test create_performance_integrator helper."""

    def test_creates_with_defaults(self) -> None:
        """Creates integrator with default values."""
        integrator = create_performance_integrator()
        assert isinstance(integrator, PerformanceEloIntegrator)

    def test_creates_with_custom_config(self) -> None:
        """Creates integrator with custom configuration."""
        integrator = create_performance_integrator(
            response_quality_weight=0.5,
            latency_weight=0.2,
            min_calls_for_adjustment=10,
        )
        assert integrator.response_quality_weight > 0
        assert integrator.min_calls_for_adjustment == 10
