"""Tests for SLA instrumentation module.

Tests the SLATracker for tracking request latency percentiles, error rates,
uptime percentages, and per-endpoint metrics.
"""

import time
import pytest

from aragora.observability.sla_instrumentation import (
    SLATracker,
    LatencyPercentiles,
    EndpointMetrics,
    get_sla_tracker,
    reset_sla_tracker,
    record_request_metric,
)


@pytest.fixture
def tracker():
    """Create a fresh SLATracker for each test."""
    return SLATracker(max_records=1000)


@pytest.fixture(autouse=True)
def _reset_global():
    """Reset the global SLA tracker between tests."""
    reset_sla_tracker()
    yield
    reset_sla_tracker()


class TestSLATrackerRecord:
    """Tests for recording request metrics."""

    def test_record_single_request(self, tracker):
        """Test recording a single request."""
        tracker.record("/api/v1/debates", "GET", 200, 0.05)

        percentiles = tracker.get_latency_percentiles()
        assert percentiles.count == 1
        assert percentiles.p50 == pytest.approx(0.05)

    def test_record_multiple_requests(self, tracker):
        """Test recording multiple requests."""
        for i in range(100):
            tracker.record("/api/v1/debates", "GET", 200, 0.01 * (i + 1))

        percentiles = tracker.get_latency_percentiles()
        assert percentiles.count == 100

    def test_record_error_request(self, tracker):
        """Test recording a 500 error request."""
        tracker.record("/api/v1/debates", "GET", 200, 0.05)
        tracker.record("/api/v1/debates", "GET", 500, 0.1)

        error_rate = tracker.get_error_rate()
        assert error_rate["total_requests"] == 2
        assert error_rate["error_count"] == 1
        assert error_rate["error_rate"] == pytest.approx(0.5)

    def test_record_evicts_old_records(self):
        """Test that old records are evicted when max is exceeded."""
        tracker = SLATracker(max_records=10)

        for i in range(20):
            tracker.record("/api/test", "GET", 200, 0.01)

        percentiles = tracker.get_latency_percentiles()
        # After eviction, we should have max_records // 2 = 5 records
        assert percentiles.count <= 10


class TestLatencyPercentiles:
    """Tests for latency percentile calculation."""

    def test_empty_tracker_returns_zeros(self, tracker):
        """Test empty tracker returns zero percentiles."""
        percentiles = tracker.get_latency_percentiles()
        assert percentiles.count == 0
        assert percentiles.p50 == 0.0
        assert percentiles.p95 == 0.0
        assert percentiles.p99 == 0.0

    def test_single_request_all_percentiles_equal(self, tracker):
        """Test single request results in all percentiles being the same."""
        tracker.record("/api/test", "GET", 200, 0.1)

        percentiles = tracker.get_latency_percentiles()
        assert percentiles.p50 == pytest.approx(0.1)
        assert percentiles.p95 == pytest.approx(0.1)
        assert percentiles.p99 == pytest.approx(0.1)

    def test_percentile_ordering(self, tracker):
        """Test that p50 <= p95 <= p99."""
        for i in range(1000):
            # Mix of fast and slow requests
            latency = 0.01 if i < 900 else 0.5
            tracker.record("/api/test", "GET", 200, latency)

        percentiles = tracker.get_latency_percentiles()
        assert percentiles.p50 <= percentiles.p95
        assert percentiles.p95 <= percentiles.p99

    def test_p99_captures_tail(self, tracker):
        """Test p99 captures tail latency from high-variance distribution."""
        # 90 fast requests, 10 slow requests - p99 should reflect slow end
        for i in range(90):
            tracker.record("/api/test", "GET", 200, 0.01)
        for i in range(10):
            tracker.record("/api/test", "GET", 200, 1.0)

        percentiles = tracker.get_latency_percentiles()
        assert percentiles.p99 >= 1.0  # Should be at the slow request value
        assert percentiles.p50 < 0.1  # Should be near the fast requests

    def test_percentiles_to_dict(self, tracker):
        """Test LatencyPercentiles.to_dict serialization."""
        tracker.record("/api/test", "GET", 200, 0.05)
        tracker.record("/api/test", "GET", 200, 0.1)

        percentiles = tracker.get_latency_percentiles()
        d = percentiles.to_dict()

        assert "p50" in d
        assert "p95" in d
        assert "p99" in d
        assert "count" in d
        assert "mean" in d
        assert "min" in d
        assert "max" in d
        assert d["count"] == 2

    def test_min_max_tracking(self, tracker):
        """Test min and max latency tracking."""
        tracker.record("/api/test", "GET", 200, 0.01)
        tracker.record("/api/test", "GET", 200, 0.5)
        tracker.record("/api/test", "GET", 200, 0.1)

        percentiles = tracker.get_latency_percentiles()
        assert percentiles.min == pytest.approx(0.01)
        assert percentiles.max == pytest.approx(0.5)

    def test_mean_calculation(self, tracker):
        """Test mean latency calculation."""
        tracker.record("/api/test", "GET", 200, 0.1)
        tracker.record("/api/test", "GET", 200, 0.2)
        tracker.record("/api/test", "GET", 200, 0.3)

        percentiles = tracker.get_latency_percentiles()
        assert percentiles.mean == pytest.approx(0.2)


class TestLatencyPercentilesByWindow:
    """Tests for windowed latency percentile queries."""

    def test_filter_by_window(self, tracker):
        """Test filtering records by time window."""
        # Record some old requests (simulate by setting window)
        tracker.record("/api/test", "GET", 200, 0.01)
        tracker.record("/api/test", "GET", 200, 0.02)

        # With a very long window, we should see all
        percentiles = tracker.get_latency_percentiles(window_seconds=86400)
        assert percentiles.count == 2

    def test_filter_by_endpoint(self, tracker):
        """Test filtering records by endpoint."""
        tracker.record("/api/debates", "GET", 200, 0.05)
        tracker.record("/api/agents", "GET", 200, 0.1)
        tracker.record("/api/debates", "GET", 200, 0.03)

        percentiles = tracker.get_latency_percentiles(endpoint="/api/debates")
        assert percentiles.count == 2

        percentiles = tracker.get_latency_percentiles(endpoint="/api/agents")
        assert percentiles.count == 1


class TestErrorRate:
    """Tests for error rate calculation."""

    def test_no_requests_zero_error_rate(self, tracker):
        """Test no requests results in 0 error rate."""
        error_rate = tracker.get_error_rate()
        assert error_rate["total_requests"] == 0
        assert error_rate["error_count"] == 0
        assert error_rate["error_rate"] == 0.0

    def test_all_success_zero_error_rate(self, tracker):
        """Test all successful requests gives 0 error rate."""
        for _ in range(10):
            tracker.record("/api/test", "GET", 200, 0.01)

        error_rate = tracker.get_error_rate()
        assert error_rate["total_requests"] == 10
        assert error_rate["error_count"] == 0
        assert error_rate["error_rate"] == 0.0

    def test_all_errors_full_error_rate(self, tracker):
        """Test all errors gives 1.0 error rate."""
        for _ in range(10):
            tracker.record("/api/test", "GET", 500, 0.01)

        error_rate = tracker.get_error_rate()
        assert error_rate["total_requests"] == 10
        assert error_rate["error_count"] == 10
        assert error_rate["error_rate"] == pytest.approx(1.0)

    def test_mixed_error_rate(self, tracker):
        """Test mixed results gives correct error rate."""
        for _ in range(90):
            tracker.record("/api/test", "GET", 200, 0.01)
        for _ in range(10):
            tracker.record("/api/test", "GET", 500, 0.01)

        error_rate = tracker.get_error_rate()
        assert error_rate["total_requests"] == 100
        assert error_rate["error_count"] == 10
        assert error_rate["error_rate"] == pytest.approx(0.1)

    def test_4xx_not_counted_as_errors(self, tracker):
        """Test that 4xx status codes are not counted as errors."""
        tracker.record("/api/test", "GET", 200, 0.01)
        tracker.record("/api/test", "GET", 400, 0.01)
        tracker.record("/api/test", "GET", 404, 0.01)

        error_rate = tracker.get_error_rate()
        assert error_rate["error_count"] == 0

    def test_error_rate_by_endpoint(self, tracker):
        """Test error rate filtered by endpoint."""
        tracker.record("/api/debates", "GET", 200, 0.01)
        tracker.record("/api/debates", "GET", 500, 0.01)
        tracker.record("/api/agents", "GET", 200, 0.01)

        debates_rate = tracker.get_error_rate(endpoint="/api/debates")
        assert debates_rate["error_rate"] == pytest.approx(0.5)

        agents_rate = tracker.get_error_rate(endpoint="/api/agents")
        assert agents_rate["error_rate"] == 0.0

    def test_error_rate_by_window(self, tracker):
        """Test error rate filtered by time window."""
        tracker.record("/api/test", "GET", 200, 0.01)
        tracker.record("/api/test", "GET", 500, 0.01)

        error_rate = tracker.get_error_rate(window_seconds=86400)
        assert error_rate["total_requests"] == 2


class TestEndpointMetrics:
    """Tests for per-endpoint metrics."""

    def test_empty_tracker_no_endpoints(self, tracker):
        """Test empty tracker returns no endpoints."""
        metrics = tracker.get_endpoint_metrics()
        assert len(metrics) == 0

    def test_single_endpoint(self, tracker):
        """Test metrics for a single endpoint."""
        tracker.record("/api/debates", "GET", 200, 0.05)
        tracker.record("/api/debates", "GET", 200, 0.1)

        metrics = tracker.get_endpoint_metrics()
        assert len(metrics) == 1
        assert metrics[0].endpoint == "/api/debates"
        assert metrics[0].total_requests == 2
        assert metrics[0].error_count == 0

    def test_multiple_endpoints(self, tracker):
        """Test metrics across multiple endpoints."""
        tracker.record("/api/debates", "GET", 200, 0.05)
        tracker.record("/api/agents", "GET", 200, 0.1)
        tracker.record("/api/knowledge", "GET", 500, 0.2)

        metrics = tracker.get_endpoint_metrics()
        assert len(metrics) == 3

        # Sorted by endpoint name
        endpoints = [m.endpoint for m in metrics]
        assert endpoints == sorted(endpoints)

    def test_endpoint_metrics_error_rate(self, tracker):
        """Test per-endpoint error rate calculation."""
        tracker.record("/api/test", "GET", 200, 0.01)
        tracker.record("/api/test", "GET", 500, 0.01)

        metrics = tracker.get_endpoint_metrics()
        assert metrics[0].error_rate == pytest.approx(0.5)
        assert metrics[0].error_count == 1

    def test_endpoint_metrics_percentiles(self, tracker):
        """Test per-endpoint latency percentiles."""
        for i in range(100):
            tracker.record("/api/test", "GET", 200, 0.01 * (i + 1))

        metrics = tracker.get_endpoint_metrics()
        assert metrics[0].latency_p50 > 0
        assert metrics[0].latency_p95 > metrics[0].latency_p50
        assert metrics[0].latency_p99 >= metrics[0].latency_p95

    def test_endpoint_metrics_to_dict(self, tracker):
        """Test EndpointMetrics.to_dict serialization."""
        tracker.record("/api/test", "GET", 200, 0.05)

        metrics = tracker.get_endpoint_metrics()
        d = metrics[0].to_dict()

        assert d["endpoint"] == "/api/test"
        assert d["total_requests"] == 1
        assert d["error_count"] == 0
        assert "latency_p50" in d
        assert "latency_p95" in d
        assert "latency_p99" in d

    def test_endpoint_metrics_windowed(self, tracker):
        """Test per-endpoint metrics with time window."""
        tracker.record("/api/test", "GET", 200, 0.01)

        metrics = tracker.get_endpoint_metrics(window_seconds=86400)
        assert len(metrics) == 1


class TestUptime:
    """Tests for uptime calculation."""

    def test_no_requests_100_percent_uptime(self, tracker):
        """Test no requests defaults to 100% uptime."""
        uptime = tracker.get_uptime()

        for period_key in ("24h", "7d", "30d"):
            assert uptime[period_key]["uptime_percent"] == 100.0

    def test_all_success_100_percent_uptime(self, tracker):
        """Test all successful requests gives 100% uptime."""
        for _ in range(100):
            tracker.record("/api/test", "GET", 200, 0.01)

        uptime = tracker.get_uptime()
        assert uptime["24h"]["uptime_percent"] == 100.0

    def test_uptime_with_errors(self, tracker):
        """Test uptime calculation with some errors."""
        for _ in range(990):
            tracker.record("/api/test", "GET", 200, 0.01)
        for _ in range(10):
            tracker.record("/api/test", "GET", 500, 0.01)

        uptime = tracker.get_uptime()
        assert uptime["24h"]["uptime_percent"] == pytest.approx(99.0)
        assert uptime["24h"]["total_requests"] == 1000
        assert uptime["24h"]["error_count"] == 10

    def test_uptime_has_all_periods(self, tracker):
        """Test uptime returns all three period windows."""
        uptime = tracker.get_uptime()
        assert "24h" in uptime
        assert "7d" in uptime
        assert "30d" in uptime

    def test_uptime_period_structure(self, tracker):
        """Test each uptime period has correct fields."""
        tracker.record("/api/test", "GET", 200, 0.01)

        uptime = tracker.get_uptime()
        for period_key in ("24h", "7d", "30d"):
            period = uptime[period_key]
            assert "uptime_percent" in period
            assert "total_requests" in period
            assert "error_count" in period
            assert "incidents" in period


class TestSLASummary:
    """Tests for the comprehensive SLA summary."""

    def test_empty_summary(self, tracker):
        """Test SLA summary with no data."""
        summary = tracker.get_sla_summary()

        assert "timestamp" in summary
        assert "latency" in summary
        assert "error_rate" in summary
        assert "uptime" in summary
        assert "endpoints" in summary
        assert "tracking_since" in summary

    def test_summary_with_data(self, tracker):
        """Test SLA summary with recorded data."""
        for i in range(50):
            tracker.record("/api/debates", "GET", 200, 0.01 * (i + 1))
        tracker.record("/api/debates", "GET", 500, 0.5)

        summary = tracker.get_sla_summary()

        assert summary["latency"]["count"] == 51
        assert summary["error_rate"]["error_count"] == 1
        assert len(summary["endpoints"]) == 1
        assert summary["endpoints"][0]["endpoint"] == "/api/debates"

    def test_summary_limits_endpoints_to_20(self, tracker):
        """Test SLA summary limits endpoint list to top 20."""
        for i in range(25):
            tracker.record(f"/api/endpoint-{i}", "GET", 200, 0.01)

        summary = tracker.get_sla_summary()
        assert len(summary["endpoints"]) <= 20


class TestReset:
    """Tests for tracker reset."""

    def test_reset_clears_data(self, tracker):
        """Test reset clears all recorded data."""
        for _ in range(10):
            tracker.record("/api/test", "GET", 200, 0.01)

        tracker.reset()

        percentiles = tracker.get_latency_percentiles()
        assert percentiles.count == 0
        error_rate = tracker.get_error_rate()
        assert error_rate["total_requests"] == 0


class TestGlobalSingleton:
    """Tests for global singleton pattern."""

    def test_get_sla_tracker_returns_tracker(self):
        """Test global getter returns a SLATracker instance."""
        tracker = get_sla_tracker()
        assert isinstance(tracker, SLATracker)

    def test_get_sla_tracker_returns_same_instance(self):
        """Test global getter returns the same instance."""
        a = get_sla_tracker()
        b = get_sla_tracker()
        assert a is b

    def test_reset_creates_new_instance(self):
        """Test reset results in new instance."""
        a = get_sla_tracker()
        reset_sla_tracker()
        b = get_sla_tracker()
        assert a is not b

    def test_record_request_metric_convenience(self):
        """Test convenience function records to global tracker."""
        record_request_metric("/api/test", "GET", 200, 0.05)

        tracker = get_sla_tracker()
        percentiles = tracker.get_latency_percentiles()
        assert percentiles.count == 1
        assert percentiles.p50 == pytest.approx(0.05)


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_recording(self, tracker):
        """Test concurrent recording does not crash."""
        import threading

        errors = []

        def record_batch(n):
            try:
                for i in range(n):
                    tracker.record("/api/test", "GET", 200, 0.01 * (i + 1))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_batch, args=(100,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        percentiles = tracker.get_latency_percentiles()
        assert percentiles.count == 400

    def test_concurrent_read_write(self, tracker):
        """Test concurrent read and write does not crash."""
        import threading

        errors = []
        running = True

        def writer():
            try:
                for i in range(200):
                    tracker.record("/api/test", "GET", 200, 0.01)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                while running:
                    tracker.get_latency_percentiles()
                    tracker.get_error_rate()
                    tracker.get_uptime()
            except Exception as e:
                errors.append(e)

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        reader_thread.start()
        writer_thread.start()
        writer_thread.join()
        running = False
        reader_thread.join()

        assert len(errors) == 0
