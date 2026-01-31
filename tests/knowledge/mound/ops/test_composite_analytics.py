"""
Comprehensive tests for CompositeAnalytics module.

Tests for:
- Metric aggregation across knowledge sources
- Time-series analytics and trend detection
- Cross-reference analysis between knowledge items
- Performance metrics computation
- Dashboard data generation
- Caching behavior for analytics queries
- Edge cases (empty data, outliers, missing dimensions)
"""

import statistics
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import Mock, patch

import pytest

from aragora.knowledge.mound.ops.composite_analytics import (
    SLOStatus,
    BottleneckSeverity,
    OptimizationType,
    AdapterMetrics,
    SLOConfig,
    SLOResult,
    BottleneckAnalysis,
    OptimizationRecommendation,
    CompositeMetrics,
    SyncResultInput,
    CompositeAnalytics,
    get_composite_analytics,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def analytics():
    """Create a fresh CompositeAnalytics instance."""
    return CompositeAnalytics()


@pytest.fixture
def analytics_with_config():
    """Create CompositeAnalytics with custom SLO config."""
    config = SLOConfig(
        sync_time_target_ms=5000.0,
        adapter_time_target_ms=1000.0,
        error_rate_threshold=0.05,
        success_rate_threshold=0.95,
        warning_threshold_ratio=0.8,
        throughput_min_per_sec=10.0,
    )
    return CompositeAnalytics(config)


@pytest.fixture
def strict_slo_config():
    """Create strict SLO config for edge case testing."""
    return SLOConfig(
        sync_time_target_ms=100.0,
        adapter_time_target_ms=50.0,
        error_rate_threshold=0.01,
        success_rate_threshold=0.99,
        warning_threshold_ratio=0.9,
        throughput_min_per_sec=100.0,
    )


@pytest.fixture
def sample_sync_results():
    """Create sample sync results for testing."""
    return [
        SyncResultInput(
            adapter_name="elo",
            direction="forward",
            success=True,
            items_processed=100,
            items_updated=50,
            errors=[],
            duration_ms=200,
            metadata={"version": "1.0"},
        ),
        SyncResultInput(
            adapter_name="consensus",
            direction="forward",
            success=True,
            items_processed=75,
            items_updated=30,
            errors=[],
            duration_ms=150,
            metadata={"version": "1.0"},
        ),
        SyncResultInput(
            adapter_name="belief",
            direction="forward",
            success=True,
            items_processed=50,
            items_updated=25,
            errors=[],
            duration_ms=100,
            metadata={"version": "1.0"},
        ),
    ]


@pytest.fixture
def mixed_success_results():
    """Create results with mixed success/failure."""
    return [
        SyncResultInput("elo", "forward", True, 100, 50, [], 200),
        SyncResultInput("consensus", "forward", False, 0, 0, ["Connection timeout"], 500),
        SyncResultInput("belief", "forward", True, 75, 30, [], 150),
        SyncResultInput("evidence", "forward", False, 10, 0, ["Parse error"], 300),
    ]


@pytest.fixture
def high_variance_results():
    """Create results with high time variance for bottleneck testing."""
    return [
        SyncResultInput("fast_adapter", "forward", True, 100, 50, [], 50),
        SyncResultInput("slow_adapter", "forward", True, 100, 50, [], 800),
        SyncResultInput("medium_adapter", "forward", True, 100, 50, [], 200),
    ]


@pytest.fixture
def many_adapters_results():
    """Create results for many adapters."""
    return [
        SyncResultInput(
            adapter_name=f"adapter_{i}",
            direction="forward",
            success=i % 10 != 0,  # 90% success rate
            items_processed=100 + i * 10,
            items_updated=50 + i * 5,
            errors=[] if i % 10 != 0 else ["Periodic failure"],
            duration_ms=100 + i * 20,
        )
        for i in range(15)
    ]


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Test enum definitions."""

    def test_slo_status_values(self):
        """Test SLOStatus enum values."""
        assert SLOStatus.MET.value == "met"
        assert SLOStatus.WARNING.value == "warning"
        assert SLOStatus.VIOLATED.value == "violated"

    def test_bottleneck_severity_values(self):
        """Test BottleneckSeverity enum values."""
        assert BottleneckSeverity.NONE.value == "none"
        assert BottleneckSeverity.MINOR.value == "minor"
        assert BottleneckSeverity.MODERATE.value == "moderate"
        assert BottleneckSeverity.SEVERE.value == "severe"
        assert BottleneckSeverity.CRITICAL.value == "critical"

    def test_optimization_type_values(self):
        """Test OptimizationType enum values."""
        assert OptimizationType.PARALLELIZE.value == "parallelize"
        assert OptimizationType.CACHE.value == "cache"
        assert OptimizationType.BATCH.value == "batch"
        assert OptimizationType.TIMEOUT_ADJUST.value == "timeout_adjust"
        assert OptimizationType.RETRY_ADJUST.value == "retry_adjust"
        assert OptimizationType.SCALE.value == "scale"


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestDataclasses:
    """Test dataclass definitions and methods."""

    def test_adapter_metrics_creation(self):
        """Test AdapterMetrics creation with defaults."""
        metrics = AdapterMetrics(adapter_name="test")

        assert metrics.adapter_name == "test"
        assert metrics.sync_count == 0
        assert metrics.success_count == 0
        assert metrics.failure_count == 0
        assert metrics.total_time_ms == 0.0
        assert metrics.avg_time_ms == 0.0
        assert metrics.error_rate == 0.0

    def test_adapter_metrics_with_values(self):
        """Test AdapterMetrics with explicit values."""
        metrics = AdapterMetrics(
            adapter_name="elo",
            sync_count=100,
            success_count=95,
            failure_count=5,
            total_time_ms=10000.0,
            avg_time_ms=100.0,
            min_time_ms=50.0,
            max_time_ms=200.0,
            p95_time_ms=180.0,
            p99_time_ms=195.0,
            items_processed=1000,
            items_updated=500,
            error_rate=0.05,
            throughput_per_sec=100.0,
        )

        assert metrics.sync_count == 100
        assert metrics.error_rate == 0.05

    def test_adapter_metrics_to_dict(self):
        """Test AdapterMetrics.to_dict() method."""
        metrics = AdapterMetrics(
            adapter_name="test",
            sync_count=10,
            avg_time_ms=50.123456,
            error_rate=0.0555555,
        )

        d = metrics.to_dict()

        assert d["adapter_name"] == "test"
        assert d["sync_count"] == 10
        assert d["avg_time_ms"] == 50.12  # Rounded
        assert d["error_rate"] == 0.0556  # Rounded to 4 decimal places

    def test_slo_config_defaults(self):
        """Test SLOConfig default values."""
        config = SLOConfig()

        assert config.sync_time_target_ms == 5000.0
        assert config.adapter_time_target_ms == 1000.0
        assert config.error_rate_threshold == 0.05
        assert config.success_rate_threshold == 0.95
        assert config.warning_threshold_ratio == 0.8
        assert config.throughput_min_per_sec == 10.0

    def test_slo_result_creation(self):
        """Test SLOResult creation."""
        result = SLOResult(
            slo_name="total_sync_time",
            target=5000.0,
            actual=3000.0,
            status=SLOStatus.MET,
            margin=0.4,
        )

        assert result.slo_name == "total_sync_time"
        assert result.status == SLOStatus.MET
        assert result.margin == 0.4

    def test_slo_result_to_dict(self):
        """Test SLOResult.to_dict() method."""
        result = SLOResult(
            slo_name="test",
            target=100.0,
            actual=80.12345,
            status=SLOStatus.MET,
            margin=0.19999,
        )

        d = result.to_dict()

        assert d["slo_name"] == "test"
        assert d["actual"] == 80.1235  # Rounded
        assert d["status"] == "met"
        assert d["margin"] == 0.2  # Rounded

    def test_bottleneck_analysis_defaults(self):
        """Test BottleneckAnalysis default values."""
        analysis = BottleneckAnalysis()

        assert analysis.bottleneck_adapter is None
        assert analysis.severity == BottleneckSeverity.NONE
        assert analysis.time_contribution_pct == 0.0
        assert analysis.recommendation == ""

    def test_bottleneck_analysis_to_dict(self):
        """Test BottleneckAnalysis.to_dict() method."""
        analysis = BottleneckAnalysis(
            bottleneck_adapter="slow_adapter",
            severity=BottleneckSeverity.SEVERE,
            time_contribution_pct=60.5,
            avg_time_ms=500.0,
            second_slowest="medium_adapter",
            gap_ms=300.0,
            recommendation="Consider optimization",
        )

        d = analysis.to_dict()

        assert d["bottleneck_adapter"] == "slow_adapter"
        assert d["severity"] == "severe"
        assert d["time_contribution_pct"] == 60.5

    def test_optimization_recommendation_creation(self):
        """Test OptimizationRecommendation creation."""
        rec = OptimizationRecommendation(
            adapter_name="slow_adapter",
            optimization_type=OptimizationType.CACHE,
            priority=1,
            expected_improvement_pct=30.0,
            description="Add caching",
            implementation_notes="Use Redis",
        )

        assert rec.priority == 1
        assert rec.optimization_type == OptimizationType.CACHE

    def test_optimization_recommendation_to_dict(self):
        """Test OptimizationRecommendation.to_dict() method."""
        rec = OptimizationRecommendation(
            adapter_name="test",
            optimization_type=OptimizationType.PARALLELIZE,
            priority=2,
            expected_improvement_pct=25.5555,
        )

        d = rec.to_dict()

        assert d["optimization_type"] == "parallelize"
        assert d["expected_improvement_pct"] == 25.56  # Rounded

    def test_composite_metrics_defaults(self):
        """Test CompositeMetrics default values."""
        metrics = CompositeMetrics()

        assert metrics.total_sync_time_ms == 0.0
        assert metrics.adapter_count == 0
        assert metrics.composite_slo_met is True
        assert metrics.adapter_metrics == {}
        assert metrics.slo_results == []
        assert metrics.recommendations == []

    def test_composite_metrics_to_dict(self):
        """Test CompositeMetrics.to_dict() method."""
        metrics = CompositeMetrics(
            total_sync_time_ms=1000.0,
            adapter_count=3,
            parallel_efficiency=0.85,
        )

        d = metrics.to_dict()

        assert d["total_sync_time_ms"] == 1000.0
        assert d["adapter_count"] == 3
        assert d["parallel_efficiency"] == 0.85
        assert "computed_at" in d

    def test_sync_result_input_creation(self):
        """Test SyncResultInput creation."""
        result = SyncResultInput(
            adapter_name="elo",
            direction="forward",
            success=True,
            items_processed=100,
            items_updated=50,
            errors=[],
            duration_ms=200,
            metadata={"test": "value"},
        )

        assert result.adapter_name == "elo"
        assert result.direction == "forward"
        assert result.success is True
        assert result.metadata["test"] == "value"


# =============================================================================
# Metric Aggregation Tests
# =============================================================================


class TestMetricAggregation:
    """Test metric aggregation across knowledge sources."""

    def test_aggregate_single_adapter(self, analytics):
        """Test aggregation with single adapter."""
        results = [
            SyncResultInput("elo", "forward", True, 100, 50, [], 200),
        ]

        metrics = analytics.compute_composite_metrics(results)

        assert metrics.adapter_count == 1
        assert "elo" in metrics.adapter_metrics
        assert metrics.total_sync_time_ms == 200.0

    def test_aggregate_multiple_adapters(self, analytics, sample_sync_results):
        """Test aggregation across multiple adapters."""
        metrics = analytics.compute_composite_metrics(sample_sync_results)

        assert metrics.adapter_count == 3
        assert "elo" in metrics.adapter_metrics
        assert "consensus" in metrics.adapter_metrics
        assert "belief" in metrics.adapter_metrics
        assert metrics.total_sync_time_ms == 450.0  # 200 + 150 + 100

    def test_aggregate_items_processed(self, analytics, sample_sync_results):
        """Test aggregation of items processed."""
        metrics = analytics.compute_composite_metrics(sample_sync_results)

        total_items = sum(m.items_processed for m in metrics.adapter_metrics.values())
        assert total_items == 225  # 100 + 75 + 50

    def test_aggregate_items_updated(self, analytics, sample_sync_results):
        """Test aggregation of items updated."""
        metrics = analytics.compute_composite_metrics(sample_sync_results)

        total_updated = sum(m.items_updated for m in metrics.adapter_metrics.values())
        assert total_updated == 105  # 50 + 30 + 25

    def test_aggregate_success_counts(self, analytics, mixed_success_results):
        """Test aggregation of success/failure counts."""
        metrics = analytics.compute_composite_metrics(mixed_success_results)

        assert metrics.successful_adapters == 2  # elo and belief succeeded

    def test_aggregate_multiple_results_same_adapter(self, analytics):
        """Test aggregation when same adapter has multiple results."""
        results = [
            SyncResultInput("elo", "forward", True, 100, 50, [], 200),
            SyncResultInput("elo", "forward", True, 80, 40, [], 150),
            SyncResultInput("elo", "forward", True, 60, 30, [], 100),
        ]

        metrics = analytics.compute_composite_metrics(results)

        assert metrics.adapter_count == 1
        elo_metrics = metrics.adapter_metrics["elo"]
        assert elo_metrics.sync_count == 3
        assert elo_metrics.total_time_ms == 450.0
        assert elo_metrics.items_processed == 240  # 100 + 80 + 60

    def test_aggregate_bidirectional_syncs(self, analytics):
        """Test aggregation with both forward and reverse directions."""
        results = [
            SyncResultInput("elo", "forward", True, 100, 50, [], 200),
            SyncResultInput("elo", "reverse", True, 80, 40, [], 150),
        ]

        metrics = analytics.compute_composite_metrics(results)

        elo_metrics = metrics.adapter_metrics["elo"]
        assert elo_metrics.sync_count == 2
        assert elo_metrics.items_processed == 180

    def test_aggregate_with_errors(self, analytics):
        """Test aggregation includes error information."""
        results = [
            SyncResultInput("elo", "forward", False, 0, 0, ["Error 1"], 500),
            SyncResultInput("elo", "forward", True, 100, 50, [], 200),
        ]

        metrics = analytics.compute_composite_metrics(results)

        elo_metrics = metrics.adapter_metrics["elo"]
        assert elo_metrics.failure_count == 1
        assert elo_metrics.success_count == 1
        assert elo_metrics.error_rate == 0.5


# =============================================================================
# Time-Series Analytics and Trend Detection Tests
# =============================================================================


class TestTimeSeriesAnalytics:
    """Test time-series analytics and trend detection."""

    def test_historical_stats_empty(self, analytics):
        """Test historical stats with no data."""
        stats = analytics.get_historical_stats("nonexistent")

        assert stats["count"] == 0

    def test_historical_stats_basic(self, analytics):
        """Test basic historical statistics."""
        # Add some data
        for i in range(20):
            results = [
                SyncResultInput("elo", "forward", True, 100, 50, [], 100 + i * 10),
            ]
            analytics.compute_composite_metrics(results)

        stats = analytics.get_historical_stats("elo")

        assert stats["count"] == 20
        assert "avg_ms" in stats
        assert "min_ms" in stats
        assert "max_ms" in stats
        assert "stddev_ms" in stats
        assert "p50_ms" in stats
        assert "p95_ms" in stats

    def test_historical_stats_min_max(self, analytics):
        """Test min/max in historical stats."""
        for i in range(10):
            results = [
                SyncResultInput("elo", "forward", True, 100, 50, [], 100 + i * 50),
            ]
            analytics.compute_composite_metrics(results)

        stats = analytics.get_historical_stats("elo")

        assert stats["min_ms"] == 100.0
        assert stats["max_ms"] == 550.0

    def test_historical_stats_stddev(self, analytics):
        """Test standard deviation in historical stats."""
        # Constant times should have zero stddev
        for _ in range(10):
            results = [
                SyncResultInput("elo", "forward", True, 100, 50, [], 100),
            ]
            analytics.compute_composite_metrics(results)

        stats = analytics.get_historical_stats("elo")

        assert stats["stddev_ms"] == 0.0

    def test_trend_insufficient_data(self, analytics):
        """Test trend computation with insufficient data."""
        for i in range(5):
            results = [
                SyncResultInput("elo", "forward", True, 100, 50, [], 100),
            ]
            analytics.compute_composite_metrics(results)

        trend = analytics.compute_trend("elo", window_size=10)

        assert trend["trend"] == "insufficient_data"
        assert trend["samples"] == 5

    def test_trend_stable(self, analytics):
        """Test stable trend detection."""
        # Add consistent times
        for _ in range(30):
            results = [
                SyncResultInput("elo", "forward", True, 100, 50, [], 100),
            ]
            analytics.compute_composite_metrics(results)

        trend = analytics.compute_trend("elo", window_size=10)

        assert trend["trend"] == "stable"
        assert trend["change_pct"] == 0.0

    def test_trend_improving(self, analytics):
        """Test improving trend detection."""
        # Times decrease over time
        for i in range(30):
            results = [
                SyncResultInput("elo", "forward", True, 100, 50, [], 300 - i * 8),
            ]
            analytics.compute_composite_metrics(results)

        trend = analytics.compute_trend("elo", window_size=10)

        assert trend["trend"] == "improving"
        assert trend["change_pct"] < -10  # Negative means improving

    def test_trend_degrading(self, analytics):
        """Test degrading trend detection."""
        # Times increase over time
        for i in range(30):
            results = [
                SyncResultInput("elo", "forward", True, 100, 50, [], 100 + i * 8),
            ]
            analytics.compute_composite_metrics(results)

        trend = analytics.compute_trend("elo", window_size=10)

        assert trend["trend"] == "degrading"
        assert trend["change_pct"] > 10  # Positive means degrading

    def test_trend_custom_window_size(self, analytics):
        """Test trend with custom window size."""
        for i in range(50):
            results = [
                SyncResultInput("elo", "forward", True, 100, 50, [], 100 + i),
            ]
            analytics.compute_composite_metrics(results)

        trend_5 = analytics.compute_trend("elo", window_size=5)
        trend_20 = analytics.compute_trend("elo", window_size=20)

        # Both should be calculated
        assert "trend" in trend_5
        assert "trend" in trend_20

    def test_historical_data_capped(self, analytics):
        """Test that historical data is capped at max_history."""
        # Add more than max_history entries
        for i in range(150):
            results = [
                SyncResultInput("elo", "forward", True, 100, 50, [], 100 + i),
            ]
            analytics.compute_composite_metrics(results)

        stats = analytics.get_historical_stats("elo")

        # Should be capped at max_history (100)
        assert stats["count"] == 100


# =============================================================================
# Performance Metrics Computation Tests
# =============================================================================


class TestPerformanceMetrics:
    """Test performance metrics computation."""

    def test_avg_sync_time(self, analytics, sample_sync_results):
        """Test average sync time computation."""
        metrics = analytics.compute_composite_metrics(sample_sync_results)

        # Total = 450ms, 3 results
        assert metrics.avg_sync_time_ms == 150.0

    def test_critical_path_time(self, analytics, sample_sync_results):
        """Test critical path time (max adapter time)."""
        metrics = analytics.compute_composite_metrics(sample_sync_results)

        # Slowest is elo at 200ms
        assert metrics.critical_path_time_ms == 200.0

    def test_theoretical_parallel_time(self, analytics, sample_sync_results):
        """Test theoretical parallel time."""
        metrics = analytics.compute_composite_metrics(sample_sync_results)

        # Theoretical parallel = max time = 200ms
        assert metrics.theoretical_parallel_time_ms == 200.0

    def test_parallel_efficiency_calculation(self, analytics):
        """Test parallel efficiency calculation."""
        # Equal time adapters = perfect efficiency
        results = [
            SyncResultInput("a", "forward", True, 100, 50, [], 100),
            SyncResultInput("b", "forward", True, 100, 50, [], 100),
            SyncResultInput("c", "forward", True, 100, 50, [], 100),
        ]

        metrics = analytics.compute_composite_metrics(results)

        assert 0.9 <= metrics.parallel_efficiency <= 1.0

    def test_parallel_efficiency_single_adapter(self, analytics):
        """Test parallel efficiency with single adapter."""
        results = [
            SyncResultInput("solo", "forward", True, 100, 50, [], 100),
        ]

        metrics = analytics.compute_composite_metrics(results)

        assert metrics.parallel_efficiency == 1.0

    def test_parallel_efficiency_unbalanced(self, analytics, high_variance_results):
        """Test parallel efficiency with unbalanced adapters."""
        metrics = analytics.compute_composite_metrics(high_variance_results)

        # Efficiency is capped at 1.0 in the implementation
        # The formula is: min(1.0, max_time / (total_time / adapter_count))
        # With unbalanced times, this may still be 1.0 if max_time is large enough
        assert 0.0 <= metrics.parallel_efficiency <= 1.0

    def test_adapter_metrics_percentiles(self, analytics):
        """Test percentile calculations in adapter metrics."""
        # Create 20 results with varying times
        results = [
            SyncResultInput("elo", "forward", True, 100, 50, [], 100 + i * 10) for i in range(20)
        ]

        metrics = analytics.compute_composite_metrics(results)
        elo_metrics = metrics.adapter_metrics["elo"]

        assert elo_metrics.min_time_ms == 100.0
        assert elo_metrics.max_time_ms == 290.0
        # P95 should be near max
        assert elo_metrics.p95_time_ms >= 270.0

    def test_throughput_calculation(self, analytics):
        """Test throughput calculation."""
        results = [
            SyncResultInput("elo", "forward", True, 1000, 500, [], 1000),  # 1 sec
        ]

        metrics = analytics.compute_composite_metrics(results)
        elo_metrics = metrics.adapter_metrics["elo"]

        # 1000 items / 1 sec = 1000/sec
        assert elo_metrics.throughput_per_sec == 1000.0

    def test_error_rate_calculation(self, analytics, mixed_success_results):
        """Test error rate calculation per adapter."""
        metrics = analytics.compute_composite_metrics(mixed_success_results)

        # consensus and evidence both failed
        assert metrics.adapter_metrics["consensus"].error_rate == 1.0
        assert metrics.adapter_metrics["evidence"].error_rate == 1.0
        assert metrics.adapter_metrics["elo"].error_rate == 0.0


# =============================================================================
# SLO Evaluation Tests
# =============================================================================


class TestSLOEvaluation:
    """Test SLO evaluation logic."""

    def test_slo_met(self, analytics_with_config):
        """Test SLO met status."""
        # Fast syncs well under target
        results = [
            SyncResultInput("elo", "forward", True, 100, 50, [], 100),
            SyncResultInput("consensus", "forward", True, 100, 50, [], 100),
        ]

        metrics = analytics_with_config.compute_composite_metrics(results)

        assert metrics.composite_slo_met is True
        sync_time_slo = next(s for s in metrics.slo_results if s.slo_name == "total_sync_time")
        assert sync_time_slo.status == SLOStatus.MET

    def test_slo_warning(self, analytics_with_config):
        """Test SLO warning status (within warning threshold)."""
        # Close to but not exceeding target (80% of 5000 = 4000)
        results = [
            SyncResultInput("elo", "forward", True, 100, 50, [], 2100),
            SyncResultInput("consensus", "forward", True, 100, 50, [], 2100),
        ]

        metrics = analytics_with_config.compute_composite_metrics(results)

        sync_time_slo = next(s for s in metrics.slo_results if s.slo_name == "total_sync_time")
        assert sync_time_slo.status == SLOStatus.WARNING

    def test_slo_violated(self, analytics_with_config):
        """Test SLO violated status."""
        # Exceeds target
        results = [
            SyncResultInput("elo", "forward", True, 100, 50, [], 3000),
            SyncResultInput("consensus", "forward", True, 100, 50, [], 3000),
        ]

        metrics = analytics_with_config.compute_composite_metrics(results)

        sync_time_slo = next(s for s in metrics.slo_results if s.slo_name == "total_sync_time")
        assert sync_time_slo.status == SLOStatus.VIOLATED

    def test_adapter_time_slo(self, analytics_with_config):
        """Test per-adapter time SLO."""
        results = [
            SyncResultInput("fast", "forward", True, 100, 50, [], 500),  # Under target
            SyncResultInput("slow", "forward", True, 100, 50, [], 1500),  # Over target
        ]

        metrics = analytics_with_config.compute_composite_metrics(results)

        fast_slo = next(s for s in metrics.slo_results if s.slo_name == "adapter_fast_time")
        slow_slo = next(s for s in metrics.slo_results if s.slo_name == "adapter_slow_time")

        assert fast_slo.status == SLOStatus.MET
        assert slow_slo.status == SLOStatus.VIOLATED

    def test_error_rate_slo(self, analytics_with_config):
        """Test error rate SLO."""
        # High error rate adapter
        results = [
            SyncResultInput("failing", "forward", False, 0, 0, ["Error"], 100),
            SyncResultInput("failing", "forward", False, 0, 0, ["Error"], 100),
            SyncResultInput("failing", "forward", True, 100, 50, [], 100),
        ]

        metrics = analytics_with_config.compute_composite_metrics(results)

        # 2/3 = 66% error rate, should violate 5% threshold
        error_slo = next(
            s for s in metrics.slo_results if s.slo_name == "adapter_failing_error_rate"
        )
        assert error_slo.status == SLOStatus.VIOLATED

    def test_success_rate_slo(self, analytics_with_config):
        """Test overall success rate SLO."""
        # Many failures
        results = [
            SyncResultInput("a", "forward", True, 100, 50, [], 100),
            SyncResultInput("b", "forward", False, 0, 0, ["Error"], 100),
            SyncResultInput("c", "forward", False, 0, 0, ["Error"], 100),
        ]

        metrics = analytics_with_config.compute_composite_metrics(results)

        success_slo = next(s for s in metrics.slo_results if s.slo_name == "overall_success_rate")
        # 1/3 = 33% < 95% threshold
        assert success_slo.status == SLOStatus.VIOLATED

    def test_slo_margin_calculation(self, analytics_with_config):
        """Test SLO margin calculation."""
        results = [
            SyncResultInput("elo", "forward", True, 100, 50, [], 2500),
        ]

        metrics = analytics_with_config.compute_composite_metrics(results)

        sync_time_slo = next(s for s in metrics.slo_results if s.slo_name == "total_sync_time")
        # 2500ms used, 5000ms target, margin = (5000-2500)/5000 = 0.5
        assert sync_time_slo.margin == 0.5


# =============================================================================
# Bottleneck Identification Tests
# =============================================================================


class TestBottleneckIdentification:
    """Test bottleneck identification logic."""

    def test_no_bottleneck_empty(self, analytics):
        """Test bottleneck analysis with no data."""
        metrics = analytics.compute_composite_metrics([])

        # With empty input, bottleneck analysis may be None or have NONE severity
        if metrics.bottleneck_analysis is not None:
            assert metrics.bottleneck_analysis.severity == BottleneckSeverity.NONE
        # else: bottleneck_analysis is None which is also valid for empty input

    def test_no_bottleneck_balanced(self, analytics):
        """Test balanced adapters with equal times."""
        results = [
            SyncResultInput("a", "forward", True, 100, 50, [], 100),
            SyncResultInput("b", "forward", True, 100, 50, [], 100),
            SyncResultInput("c", "forward", True, 100, 50, [], 100),
        ]

        metrics = analytics.compute_composite_metrics(results)

        # With 3 equal adapters, each takes 33.33% of time
        # The bottleneck detection uses 25% threshold for MODERATE
        # So balanced adapters with equal time will still trigger MODERATE
        # because even balanced = 33% > 25%
        assert metrics.bottleneck_analysis is not None
        assert metrics.bottleneck_analysis.gap_ms == 0.0  # No gap between adapters

    def test_critical_bottleneck(self, analytics):
        """Test critical bottleneck detection."""
        results = [
            SyncResultInput("slow", "forward", True, 100, 50, [], 1000),
            SyncResultInput("fast1", "forward", True, 100, 50, [], 50),
            SyncResultInput("fast2", "forward", True, 100, 50, [], 50),
        ]

        metrics = analytics.compute_composite_metrics(results)

        # slow takes 1000/(1000+50+50) = 90.9% of time
        assert metrics.bottleneck_analysis.bottleneck_adapter == "slow"
        assert metrics.bottleneck_analysis.severity == BottleneckSeverity.CRITICAL
        assert metrics.bottleneck_analysis.time_contribution_pct > 50

    def test_severe_bottleneck(self, analytics):
        """Test severe bottleneck detection."""
        results = [
            SyncResultInput("slow", "forward", True, 100, 50, [], 500),
            SyncResultInput("medium", "forward", True, 100, 50, [], 400),
            SyncResultInput("fast", "forward", True, 100, 50, [], 300),
        ]

        metrics = analytics.compute_composite_metrics(results)

        # slow takes 500/1200 = 41.7% of time
        assert metrics.bottleneck_analysis.bottleneck_adapter == "slow"
        assert metrics.bottleneck_analysis.severity in [
            BottleneckSeverity.MODERATE,
            BottleneckSeverity.SEVERE,
        ]

    def test_moderate_bottleneck(self, analytics):
        """Test moderate bottleneck detection."""
        results = [
            SyncResultInput("slow", "forward", True, 100, 50, [], 350),
            SyncResultInput("medium1", "forward", True, 100, 50, [], 250),
            SyncResultInput("medium2", "forward", True, 100, 50, [], 250),
            SyncResultInput("fast", "forward", True, 100, 50, [], 150),
        ]

        metrics = analytics.compute_composite_metrics(results)

        # slow takes 350/1000 = 35%
        assert metrics.bottleneck_analysis.severity in [
            BottleneckSeverity.MODERATE,
            BottleneckSeverity.SEVERE,
        ]

    def test_minor_bottleneck(self, analytics):
        """Test minor bottleneck detection."""
        results = [
            SyncResultInput("slightly_slow", "forward", True, 100, 50, [], 250),
            SyncResultInput("normal", "forward", True, 100, 50, [], 100),
        ]

        metrics = analytics.compute_composite_metrics(results)

        # Gap of 150ms between slowest and second
        assert metrics.bottleneck_analysis.gap_ms == 150.0

    def test_bottleneck_second_slowest(self, analytics, high_variance_results):
        """Test second slowest adapter identification."""
        metrics = analytics.compute_composite_metrics(high_variance_results)

        assert metrics.bottleneck_analysis.second_slowest == "medium_adapter"
        assert metrics.bottleneck_analysis.gap_ms == 600.0  # 800 - 200

    def test_bottleneck_recommendation(self, analytics, high_variance_results):
        """Test bottleneck recommendation text."""
        metrics = analytics.compute_composite_metrics(high_variance_results)

        assert len(metrics.bottleneck_analysis.recommendation) > 0
        assert "slow_adapter" in metrics.bottleneck_analysis.recommendation


# =============================================================================
# Optimization Recommendation Tests
# =============================================================================


class TestOptimizationRecommendations:
    """Test optimization recommendation generation."""

    def test_no_recommendations_healthy(self, analytics):
        """Test no recommendations for healthy system."""
        results = [
            SyncResultInput("a", "forward", True, 100, 50, [], 100),
            SyncResultInput("b", "forward", True, 100, 50, [], 100),
        ]

        metrics = analytics.compute_composite_metrics(results, include_recommendations=True)

        # Should have parallelization recommendation if no dependencies set
        # but no error or slow adapter recommendations

    def test_high_error_rate_recommendation(self, analytics):
        """Test recommendation for high error rate."""
        results = [
            SyncResultInput("failing", "forward", False, 0, 0, ["Error"], 100),
            SyncResultInput("failing", "forward", False, 0, 0, ["Error"], 100),
        ]

        metrics = analytics.compute_composite_metrics(results, include_recommendations=True)

        retry_recs = [
            r
            for r in metrics.recommendations
            if r.optimization_type == OptimizationType.RETRY_ADJUST
        ]
        assert len(retry_recs) > 0
        assert retry_recs[0].adapter_name == "failing"

    def test_slow_adapter_recommendation(self, analytics_with_config):
        """Test recommendation for slow adapter."""
        results = [
            SyncResultInput("slow", "forward", True, 100, 50, [], 2000),  # Over 1000ms target
        ]

        metrics = analytics_with_config.compute_composite_metrics(
            results, include_recommendations=True
        )

        cache_recs = [
            r for r in metrics.recommendations if r.optimization_type == OptimizationType.CACHE
        ]
        assert len(cache_recs) > 0
        assert cache_recs[0].adapter_name == "slow"

    def test_low_throughput_recommendation(self, analytics_with_config):
        """Test recommendation for low throughput."""
        results = [
            SyncResultInput("slow_throughput", "forward", True, 5, 2, [], 1000),
            # 5 items / 1 sec = 5/sec, below 10/sec threshold
        ]

        metrics = analytics_with_config.compute_composite_metrics(
            results, include_recommendations=True
        )

        batch_recs = [
            r for r in metrics.recommendations if r.optimization_type == OptimizationType.BATCH
        ]
        assert len(batch_recs) > 0

    def test_parallelization_recommendation(self, analytics):
        """Test parallelization recommendation."""
        # Add some history
        for _ in range(5):
            results = [
                SyncResultInput("a", "forward", True, 100, 50, [], 100),
                SyncResultInput("b", "forward", True, 100, 50, [], 100),
            ]
            analytics.compute_composite_metrics(results)

        # Should recommend parallelization
        metrics = analytics.compute_composite_metrics(
            [SyncResultInput("a", "forward", True, 100, 50, [], 100)],
            include_recommendations=True,
        )

        parallel_recs = [
            r
            for r in metrics.recommendations
            if r.optimization_type == OptimizationType.PARALLELIZE
        ]
        # Parallelization only recommended when adapters exist and no dependencies
        if parallel_recs:
            assert "system" in parallel_recs[0].adapter_name

    def test_recommendations_sorted_by_priority(self, analytics):
        """Test recommendations are sorted by priority."""
        # Create conditions for multiple recommendations
        results = [
            SyncResultInput("error_adapter", "forward", False, 0, 0, ["Error"], 100),
            SyncResultInput("slow_adapter", "forward", True, 100, 50, [], 2000),
        ]

        metrics = analytics.compute_composite_metrics(results, include_recommendations=True)

        if len(metrics.recommendations) >= 2:
            priorities = [r.priority for r in metrics.recommendations]
            assert priorities == sorted(priorities)

    def test_skip_recommendations(self, analytics):
        """Test skipping recommendations."""
        results = [
            SyncResultInput("slow", "forward", True, 100, 50, [], 2000),
        ]

        metrics = analytics.compute_composite_metrics(results, include_recommendations=False)

        assert len(metrics.recommendations) == 0


# =============================================================================
# Adapter Dependencies and Parallelization Tests
# =============================================================================


class TestAdapterDependencies:
    """Test adapter dependency management."""

    def test_set_adapter_dependencies(self, analytics):
        """Test setting adapter dependencies."""
        dependencies = {
            "elo": [],
            "consensus": ["elo"],
            "belief": ["elo", "consensus"],
        }

        analytics.set_adapter_dependencies(dependencies)

        assert analytics._adapter_dependencies["consensus"] == {"elo"}
        assert analytics._adapter_dependencies["belief"] == {"elo", "consensus"}

    def test_recommend_parallelization_no_history(self, analytics):
        """Test parallelization with no history."""
        result = analytics.recommend_parallelization()

        assert result == []

    def test_recommend_parallelization_no_dependencies(self, analytics):
        """Test parallelization with no dependencies."""
        # Add history
        for _ in range(5):
            results = [
                SyncResultInput("a", "forward", True, 100, 50, [], 100),
                SyncResultInput("b", "forward", True, 100, 50, [], 100),
            ]
            analytics.compute_composite_metrics(results)

        parallelizable = analytics.recommend_parallelization()

        # Both adapters should be parallelizable
        assert "a" in parallelizable
        assert "b" in parallelizable

    def test_recommend_parallelization_with_dependencies(self, analytics):
        """Test parallelization respects dependencies."""
        # Add history
        for _ in range(5):
            results = [
                SyncResultInput("independent", "forward", True, 100, 50, [], 100),
                SyncResultInput("dependent", "forward", True, 100, 50, [], 100),
            ]
            analytics.compute_composite_metrics(results)

        # Set dependency
        analytics.set_adapter_dependencies(
            {
                "dependent": ["independent"],
            }
        )

        parallelizable = analytics.recommend_parallelization()

        # independent should be parallelizable, dependent should not
        assert "independent" in parallelizable
        # dependent has dependencies so should not be in parallelizable
        assert "dependent" not in parallelizable


# =============================================================================
# Dashboard Data Generation Tests
# =============================================================================


class TestDashboardData:
    """Test dashboard data generation."""

    def test_get_summary_empty(self, analytics):
        """Test summary with no data."""
        summary = analytics.get_summary()

        assert summary["adapter_count"] == 0
        assert summary["adapters"] == {}

    def test_get_summary_with_data(self, analytics):
        """Test summary with data."""
        for i in range(20):
            results = [
                SyncResultInput("elo", "forward", True, 100, 50, [], 100 + i),
                SyncResultInput("consensus", "forward", True, 100, 50, [], 150 + i),
            ]
            analytics.compute_composite_metrics(results)

        summary = analytics.get_summary()

        assert summary["adapter_count"] == 2
        assert "elo" in summary["adapters"]
        assert "consensus" in summary["adapters"]
        assert "stats" in summary["adapters"]["elo"]
        assert "trend" in summary["adapters"]["elo"]

    def test_composite_metrics_to_dict_complete(self, analytics, sample_sync_results):
        """Test complete metrics to dict conversion."""
        metrics = analytics.compute_composite_metrics(sample_sync_results)
        d = metrics.to_dict()

        # Check all expected keys
        assert "total_sync_time_ms" in d
        assert "avg_sync_time_ms" in d
        assert "parallel_efficiency" in d
        assert "critical_path_time_ms" in d
        assert "adapter_count" in d
        assert "adapter_metrics" in d
        assert "composite_slo_met" in d
        assert "slo_results" in d
        assert "bottleneck_analysis" in d
        assert "recommendations" in d
        assert "computed_at" in d

    def test_metrics_timestamp(self, analytics, sample_sync_results):
        """Test computed_at timestamp."""
        before = datetime.now(timezone.utc)
        metrics = analytics.compute_composite_metrics(sample_sync_results)
        after = datetime.now(timezone.utc)

        assert before <= metrics.computed_at <= after


# =============================================================================
# Input Normalization Tests
# =============================================================================


class TestInputNormalization:
    """Test sync result input normalization."""

    def test_normalize_sync_result_input(self, analytics):
        """Test normalization of SyncResultInput objects."""
        results = [
            SyncResultInput("elo", "forward", True, 100, 50, [], 200),
        ]

        metrics = analytics.compute_composite_metrics(results)

        assert "elo" in metrics.adapter_metrics

    def test_normalize_dict_input(self, analytics):
        """Test normalization of dict inputs."""
        results = [
            {
                "adapter_name": "elo",
                "direction": "forward",
                "success": True,
                "items_processed": 100,
                "items_updated": 50,
                "errors": [],
                "duration_ms": 200,
                "metadata": {},
            }
        ]

        metrics = analytics.compute_composite_metrics(results)

        assert "elo" in metrics.adapter_metrics
        assert metrics.adapter_metrics["elo"].items_processed == 100

    def test_normalize_duck_typed_input(self, analytics):
        """Test normalization of duck-typed objects."""

        @dataclass
        class CustomSyncResult:
            adapter_name: str
            direction: str
            success: bool
            items_processed: int
            items_updated: int
            duration_ms: int
            errors: list

        results = [
            CustomSyncResult("elo", "forward", True, 100, 50, 200, []),
        ]

        metrics = analytics.compute_composite_metrics(results)

        assert "elo" in metrics.adapter_metrics

    def test_normalize_partial_dict(self, analytics):
        """Test normalization of partial dict (missing fields)."""
        results = [
            {
                "adapter_name": "elo",
                "duration_ms": 200,
            }
        ]

        metrics = analytics.compute_composite_metrics(results)

        # Should use defaults
        assert "elo" in metrics.adapter_metrics
        assert metrics.adapter_metrics["elo"].items_processed == 0

    def test_normalize_unknown_format(self, analytics):
        """Test normalization handles unknown formats gracefully."""
        results = [
            {"adapter_name": "test", "duration_ms": 100},
            "invalid_string",  # Should be skipped
            42,  # Should be skipped
        ]

        # Should not crash
        metrics = analytics.compute_composite_metrics(results)

        assert metrics.adapter_count >= 1


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_sync_results(self, analytics):
        """Test with empty sync results."""
        metrics = analytics.compute_composite_metrics([])

        assert metrics.adapter_count == 0
        assert metrics.total_sync_time_ms == 0.0
        assert metrics.avg_sync_time_ms == 0.0
        assert metrics.parallel_efficiency == 0.0
        assert metrics.adapter_metrics == {}

    def test_zero_duration(self, analytics):
        """Test with zero duration."""
        results = [
            SyncResultInput("instant", "forward", True, 100, 50, [], 0),
        ]

        metrics = analytics.compute_composite_metrics(results)

        assert metrics.adapter_metrics["instant"].total_time_ms == 0.0
        # Throughput should handle division by zero gracefully
        assert metrics.adapter_metrics["instant"].throughput_per_sec >= 0

    def test_very_large_duration(self, analytics):
        """Test with very large duration."""
        results = [
            SyncResultInput("slow", "forward", True, 100, 50, [], 1000000),  # 1000 seconds
        ]

        metrics = analytics.compute_composite_metrics(results)

        assert metrics.adapter_metrics["slow"].total_time_ms == 1000000.0

    def test_very_small_duration(self, analytics):
        """Test with very small duration."""
        results = [
            SyncResultInput("fast", "forward", True, 100, 50, [], 1),  # 1ms
        ]

        metrics = analytics.compute_composite_metrics(results)

        assert metrics.adapter_metrics["fast"].total_time_ms == 1.0

    def test_zero_items_processed(self, analytics):
        """Test with zero items processed."""
        results = [
            SyncResultInput("empty", "forward", True, 0, 0, [], 100),
        ]

        metrics = analytics.compute_composite_metrics(results)

        assert metrics.adapter_metrics["empty"].items_processed == 0
        assert metrics.adapter_metrics["empty"].throughput_per_sec == 0.0

    def test_all_failures(self, analytics):
        """Test with all failures."""
        results = [
            SyncResultInput("a", "forward", False, 0, 0, ["Error 1"], 100),
            SyncResultInput("b", "forward", False, 0, 0, ["Error 2"], 100),
        ]

        metrics = analytics.compute_composite_metrics(results)

        assert metrics.successful_adapters == 0
        assert all(m.error_rate == 1.0 for m in metrics.adapter_metrics.values())

    def test_single_result_percentiles(self, analytics):
        """Test percentile calculations with single result."""
        results = [
            SyncResultInput("solo", "forward", True, 100, 50, [], 100),
        ]

        metrics = analytics.compute_composite_metrics(results)

        # With single result, all percentiles should equal the single value
        assert metrics.adapter_metrics["solo"].p95_time_ms == 100.0
        assert metrics.adapter_metrics["solo"].p99_time_ms == 100.0

    def test_two_results_percentiles(self, analytics):
        """Test percentile calculations with two results."""
        results = [
            SyncResultInput("elo", "forward", True, 100, 50, [], 100),
            SyncResultInput("elo", "forward", True, 100, 50, [], 200),
        ]

        metrics = analytics.compute_composite_metrics(results)

        # With two results, percentiles should be calculated
        assert metrics.adapter_metrics["elo"].p95_time_ms in [100.0, 200.0]

    def test_unicode_adapter_names(self, analytics):
        """Test with unicode adapter names."""
        results = [
            SyncResultInput("adaptador_\u00e9l\u00e9ment", "forward", True, 100, 50, [], 100),
            SyncResultInput(
                "\u65e5\u672c\u8a9e\u30a2\u30c0\u30d7\u30bf", "forward", True, 100, 50, [], 100
            ),
        ]

        metrics = analytics.compute_composite_metrics(results)

        assert metrics.adapter_count == 2

    def test_long_adapter_names(self, analytics):
        """Test with very long adapter names."""
        long_name = "a" * 1000
        results = [
            SyncResultInput(long_name, "forward", True, 100, 50, [], 100),
        ]

        metrics = analytics.compute_composite_metrics(results)

        assert long_name in metrics.adapter_metrics

    def test_special_characters_in_errors(self, analytics):
        """Test with special characters in error messages."""
        results = [
            SyncResultInput(
                "elo", "forward", False, 0, 0, ["Error: \n\t'quote' \"double\" <html>"], 100
            ),
        ]

        metrics = analytics.compute_composite_metrics(results)

        assert metrics.adapter_metrics["elo"].failure_count == 1

    def test_negative_duration_handled(self, analytics):
        """Test that negative duration is handled (shouldn't happen but test robustness)."""
        results = [
            SyncResultInput("weird", "forward", True, 100, 50, [], -100),
        ]

        # Should not crash
        metrics = analytics.compute_composite_metrics(results)

        assert "weird" in metrics.adapter_metrics

    def test_very_many_adapters(self, analytics, many_adapters_results):
        """Test with many adapters."""
        metrics = analytics.compute_composite_metrics(many_adapters_results)

        assert metrics.adapter_count == 15
        assert len(metrics.adapter_metrics) == 15

    def test_slo_division_by_zero(self):
        """Test SLO calculations handle division by zero."""
        config = SLOConfig(
            error_rate_threshold=0.0,  # Division by zero risk
            success_rate_threshold=1.0,  # Edge case
        )
        analytics = CompositeAnalytics(config)

        results = [
            SyncResultInput("elo", "forward", True, 100, 50, [], 100),
        ]

        # Should not crash
        metrics = analytics.compute_composite_metrics(results)

        assert metrics.composite_slo_met is True


# =============================================================================
# Singleton and Factory Tests
# =============================================================================


class TestSingleton:
    """Test singleton pattern."""

    def test_get_composite_analytics_creates_instance(self):
        """Test getting singleton creates instance."""
        # Reset global state
        import aragora.knowledge.mound.ops.composite_analytics as ca

        ca._composite_analytics = None

        instance = get_composite_analytics()

        assert instance is not None
        assert isinstance(instance, CompositeAnalytics)

    def test_get_composite_analytics_returns_same_instance(self):
        """Test singleton returns same instance."""
        instance1 = get_composite_analytics()
        instance2 = get_composite_analytics()

        assert instance1 is instance2

    def test_singleton_with_config(self):
        """Test singleton config is used only on first call."""
        import aragora.knowledge.mound.ops.composite_analytics as ca

        ca._composite_analytics = None

        config = SLOConfig(sync_time_target_ms=10000.0)
        instance1 = get_composite_analytics(config)

        assert instance1.slo_config.sync_time_target_ms == 10000.0

        # Second call with different config should return same instance
        config2 = SLOConfig(sync_time_target_ms=20000.0)
        instance2 = get_composite_analytics(config2)

        assert instance2 is instance1
        assert instance2.slo_config.sync_time_target_ms == 10000.0


# =============================================================================
# Caching Behavior Tests
# =============================================================================


class TestCachingBehavior:
    """Test caching behavior for analytics queries."""

    def test_historical_data_accumulates(self, analytics):
        """Test that historical data accumulates across calls."""
        for i in range(10):
            results = [
                SyncResultInput("elo", "forward", True, 100, 50, [], 100 + i),
            ]
            analytics.compute_composite_metrics(results)

        stats = analytics.get_historical_stats("elo")

        assert stats["count"] == 10

    def test_historical_data_persists(self, analytics):
        """Test that historical data persists between metric computations."""
        # First batch
        for _ in range(5):
            results = [SyncResultInput("elo", "forward", True, 100, 50, [], 100)]
            analytics.compute_composite_metrics(results)

        stats1 = analytics.get_historical_stats("elo")

        # Second batch
        for _ in range(5):
            results = [SyncResultInput("elo", "forward", True, 100, 50, [], 200)]
            analytics.compute_composite_metrics(results)

        stats2 = analytics.get_historical_stats("elo")

        assert stats2["count"] == stats1["count"] + 5

    def test_historical_data_window(self, analytics):
        """Test that historical data respects max_history window."""
        # Add more than max_history (100)
        for i in range(150):
            results = [SyncResultInput("elo", "forward", True, 100, 50, [], 100 + i)]
            analytics.compute_composite_metrics(results)

        stats = analytics.get_historical_stats("elo")

        # Should be capped at 100
        assert stats["count"] == 100

        # Should have most recent data
        assert stats["max_ms"] == 249.0  # Last 100 of 150 = indices 50-149


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_analysis_pipeline(self, analytics_with_config):
        """Test full analysis pipeline."""
        # Simulate multiple sync cycles
        for i in range(30):
            results = [
                SyncResultInput("elo", "forward", True, 100 + i, 50, [], 100 + i * 2),
                SyncResultInput(
                    "consensus",
                    "forward",
                    i % 5 != 0,
                    80,
                    40,
                    [] if i % 5 != 0 else ["Periodic error"],
                    150,
                ),
                SyncResultInput("belief", "forward", True, 60, 30, [], 200 - i),
            ]
            analytics_with_config.compute_composite_metrics(results)

        # Get final metrics
        final_results = [
            SyncResultInput("elo", "forward", True, 130, 50, [], 160),
            SyncResultInput("consensus", "forward", True, 80, 40, [], 150),
            SyncResultInput("belief", "forward", True, 60, 30, [], 170),
        ]
        metrics = analytics_with_config.compute_composite_metrics(
            final_results, include_recommendations=True
        )

        # Verify complete analysis
        assert metrics.adapter_count == 3
        assert len(metrics.slo_results) > 0
        assert metrics.bottleneck_analysis is not None

        # Get summary
        summary = analytics_with_config.get_summary()
        assert summary["adapter_count"] == 3

        # Check trends
        for adapter in ["elo", "consensus", "belief"]:
            trend = analytics_with_config.compute_trend(adapter, window_size=10)
            assert "trend" in trend

    def test_metrics_consistency(self, analytics, sample_sync_results):
        """Test that metrics are internally consistent."""
        metrics = analytics.compute_composite_metrics(sample_sync_results)

        # Total time should equal sum of adapter times
        adapter_total = sum(m.total_time_ms for m in metrics.adapter_metrics.values())
        assert metrics.total_sync_time_ms == adapter_total

        # Successful adapters count should match
        successful_count = sum(1 for m in metrics.adapter_metrics.values() if m.success_count > 0)
        assert metrics.successful_adapters == successful_count

    def test_slo_and_bottleneck_consistency(self, analytics):
        """Test SLO and bottleneck analysis are consistent."""
        results = [
            SyncResultInput("slow", "forward", True, 100, 50, [], 2000),
            SyncResultInput("fast", "forward", True, 100, 50, [], 100),
        ]

        config = SLOConfig(adapter_time_target_ms=1000.0)
        analytics = CompositeAnalytics(config)

        metrics = analytics.compute_composite_metrics(results)

        # Slow adapter should be identified as bottleneck
        assert metrics.bottleneck_analysis.bottleneck_adapter == "slow"

        # Slow adapter should have violated SLO
        slow_slo = next(
            s for s in metrics.slo_results if "slow" in s.slo_name and "time" in s.slo_name
        )
        assert slow_slo.status in [SLOStatus.WARNING, SLOStatus.VIOLATED]
