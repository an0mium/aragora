"""
Tests for SLO Enforcer - real-time metric tracking and enforcement.

Tests the SLOEnforcer class that tracks individual request metrics,
computes SLO compliance, detects violations, and provides error budgets.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from aragora.observability.slo import (
    DEFAULT_AVAILABILITY_TARGET,
    DEFAULT_ERROR_RATE_TARGET,
    DEFAULT_LATENCY_P95_MS,
    DEFAULT_LATENCY_P99_MS,
    SLOEnforcer,
    SLOViolation,
    get_slo_enforcer,
    reset_slo_enforcer,
)


class TestSLOEnforcerInit:
    """Tests for SLOEnforcer initialization."""

    def test_default_init(self):
        """Enforcer initializes with default window of 1 hour."""
        enforcer = SLOEnforcer()
        assert enforcer._window_seconds == 3600.0
        assert enforcer._requests == []
        assert enforcer._violations == []

    def test_custom_window(self):
        """Enforcer accepts custom window duration."""
        enforcer = SLOEnforcer(window_seconds=300.0)
        assert enforcer._window_seconds == 300.0

    def test_with_violation_callback(self):
        """Enforcer accepts a violation callback."""
        callback = MagicMock()
        enforcer = SLOEnforcer(on_violation=callback)
        assert enforcer._on_violation is callback


class TestSLOEnforcerRecordRequest:
    """Tests for recording requests."""

    def test_record_successful_request(self):
        """Successful request is recorded correctly."""
        enforcer = SLOEnforcer()
        enforcer.record_request(latency_ms=50.0, success=True)

        assert len(enforcer._requests) == 1
        req = enforcer._requests[0]
        assert req["latency_ms"] == 50.0
        assert req["success"] is True
        assert req["endpoint"] is None

    def test_record_failed_request(self):
        """Failed request is recorded correctly."""
        enforcer = SLOEnforcer()
        enforcer.record_request(latency_ms=150.0, success=False)

        assert len(enforcer._requests) == 1
        assert enforcer._requests[0]["success"] is False

    def test_record_with_endpoint(self):
        """Request records the endpoint path."""
        enforcer = SLOEnforcer()
        enforcer.record_request(latency_ms=30.0, success=True, endpoint="/api/v1/health")

        assert enforcer._requests[0]["endpoint"] == "/api/v1/health"

    def test_multiple_requests(self):
        """Multiple requests are tracked."""
        enforcer = SLOEnforcer()
        for i in range(100):
            enforcer.record_request(latency_ms=float(i), success=True)

        assert len(enforcer._requests) == 100

    def test_prune_old_requests(self):
        """Requests outside the rolling window are pruned."""
        enforcer = SLOEnforcer(window_seconds=60.0)

        # Record a request and manually backdate it
        enforcer.record_request(latency_ms=10.0, success=True)
        enforcer._requests[0]["timestamp"] = datetime.now(timezone.utc) - timedelta(
            seconds=120
        )

        # Record a new request - this triggers pruning
        enforcer.record_request(latency_ms=20.0, success=True)

        assert len(enforcer._requests) == 1
        assert enforcer._requests[0]["latency_ms"] == 20.0


class TestSLOEnforcerPercentile:
    """Tests for percentile calculation."""

    def test_percentile_empty(self):
        """Percentile of empty list returns 0."""
        enforcer = SLOEnforcer()
        assert enforcer._percentile([], 50) == 0.0

    def test_percentile_single(self):
        """Percentile of single value returns that value."""
        enforcer = SLOEnforcer()
        assert enforcer._percentile([42.0], 50) == 42.0
        assert enforcer._percentile([42.0], 99) == 42.0

    def test_percentile_p50(self):
        """p50 returns the median value."""
        enforcer = SLOEnforcer()
        values = list(range(1, 101))  # 1 to 100
        p50 = enforcer._percentile(values, 50)
        assert 49.0 <= p50 <= 51.0

    def test_percentile_p95(self):
        """p95 returns the 95th percentile."""
        enforcer = SLOEnforcer()
        values = list(range(1, 101))
        p95 = enforcer._percentile(values, 95)
        assert 94.0 <= p95 <= 96.0

    def test_percentile_p99(self):
        """p99 returns the 99th percentile."""
        enforcer = SLOEnforcer()
        values = list(range(1, 101))
        p99 = enforcer._percentile(values, 99)
        assert 98.0 <= p99 <= 100.0


class TestSLOEnforcerGetMetrics:
    """Tests for aggregated metrics computation."""

    def test_no_requests(self):
        """Metrics with no requests return zero values."""
        enforcer = SLOEnforcer()
        metrics = enforcer.get_metrics()

        assert metrics["total_requests"] == 0
        assert metrics["successful_requests"] == 0
        assert metrics["failed_requests"] == 0
        assert metrics["latency_p50_ms"] == 0.0
        assert metrics["latency_p95_ms"] == 0.0
        assert metrics["latency_p99_ms"] == 0.0
        assert metrics["error_rate"] == 0.0
        assert metrics["uptime"] == 1.0
        assert metrics["throughput_rps"] == 0.0

    def test_all_successful(self):
        """Metrics with all successful requests show 0 error rate."""
        enforcer = SLOEnforcer()
        for i in range(100):
            enforcer.record_request(latency_ms=float(i + 1), success=True)

        metrics = enforcer.get_metrics()
        assert metrics["total_requests"] == 100
        assert metrics["successful_requests"] == 100
        assert metrics["failed_requests"] == 0
        assert metrics["error_rate"] == 0.0
        assert metrics["uptime"] == 1.0

    def test_mixed_results(self):
        """Metrics correctly compute error rate with mixed results."""
        enforcer = SLOEnforcer()
        # 95 successful, 5 failed = 5% error rate
        for _ in range(95):
            enforcer.record_request(latency_ms=50.0, success=True)
        for _ in range(5):
            enforcer.record_request(latency_ms=500.0, success=False)

        metrics = enforcer.get_metrics()
        assert metrics["total_requests"] == 100
        assert metrics["successful_requests"] == 95
        assert metrics["failed_requests"] == 5
        assert abs(metrics["error_rate"] - 0.05) < 0.001
        assert abs(metrics["uptime"] - 0.95) < 0.001

    def test_latency_percentiles(self):
        """Metrics compute latency percentiles correctly."""
        enforcer = SLOEnforcer()
        # Most requests fast, a few slow
        for _ in range(90):
            enforcer.record_request(latency_ms=50.0, success=True)
        for _ in range(5):
            enforcer.record_request(latency_ms=400.0, success=True)
        for _ in range(5):
            enforcer.record_request(latency_ms=1500.0, success=True)

        metrics = enforcer.get_metrics()
        assert metrics["latency_p50_ms"] <= 100.0  # Median should be near 50
        assert metrics["latency_p95_ms"] > 50.0  # p95 captures the slow ones
        assert metrics["latency_p99_ms"] > 100.0  # p99 captures the slowest

    def test_throughput(self):
        """Metrics compute throughput as requests per second."""
        enforcer = SLOEnforcer()
        for _ in range(100):
            enforcer.record_request(latency_ms=10.0, success=True)

        metrics = enforcer.get_metrics()
        assert metrics["throughput_rps"] > 0


class TestSLOEnforcerCheckViolations:
    """Tests for SLO violation detection."""

    def test_no_violations_when_all_good(self):
        """No violations when all metrics are within targets."""
        enforcer = SLOEnforcer()
        # All fast, all successful
        for _ in range(1000):
            enforcer.record_request(latency_ms=50.0, success=True)

        violations = enforcer.check_violations()
        assert violations == []

    def test_no_violations_empty(self):
        """No violations when no requests recorded."""
        enforcer = SLOEnforcer()
        violations = enforcer.check_violations()
        assert violations == []

    def test_latency_p95_violation(self):
        """Violation detected when p95 latency exceeds 500ms."""
        enforcer = SLOEnforcer()
        # Most requests are slow enough to push p95 above 500ms
        for _ in range(100):
            enforcer.record_request(latency_ms=600.0, success=True)

        violations = enforcer.check_violations()
        violation_names = [v.slo_name for v in violations]
        assert "latency_p95" in violation_names

    def test_latency_p99_violation(self):
        """Violation detected when p99 latency exceeds 2000ms."""
        enforcer = SLOEnforcer()
        for _ in range(100):
            enforcer.record_request(latency_ms=2500.0, success=True)

        violations = enforcer.check_violations()
        violation_names = [v.slo_name for v in violations]
        assert "latency_p99" in violation_names

    def test_error_rate_violation(self):
        """Violation detected when error rate exceeds 1%."""
        enforcer = SLOEnforcer()
        # 5% error rate
        for _ in range(95):
            enforcer.record_request(latency_ms=50.0, success=True)
        for _ in range(5):
            enforcer.record_request(latency_ms=50.0, success=False)

        violations = enforcer.check_violations()
        violation_names = [v.slo_name for v in violations]
        assert "error_rate" in violation_names

    def test_availability_violation(self):
        """Violation detected when uptime falls below 99.9%."""
        enforcer = SLOEnforcer()
        # 99% uptime (below 99.9% target)
        for _ in range(990):
            enforcer.record_request(latency_ms=50.0, success=True)
        for _ in range(10):
            enforcer.record_request(latency_ms=50.0, success=False)

        violations = enforcer.check_violations()
        violation_names = [v.slo_name for v in violations]
        assert "availability" in violation_names

    def test_violation_callback_invoked(self):
        """Violation callback is invoked when violations occur."""
        callback = MagicMock()
        enforcer = SLOEnforcer(on_violation=callback)

        for _ in range(100):
            enforcer.record_request(latency_ms=600.0, success=True)

        violations = enforcer.check_violations()
        assert len(violations) > 0
        assert callback.call_count >= 1

    def test_violations_stored(self):
        """Violations are accumulated in the enforcer."""
        enforcer = SLOEnforcer()
        for _ in range(100):
            enforcer.record_request(latency_ms=600.0, success=True)

        enforcer.check_violations()
        assert len(enforcer._violations) > 0

    def test_multiple_violations_at_once(self):
        """Multiple SLOs can be violated simultaneously."""
        enforcer = SLOEnforcer()
        # Slow AND high error rate
        for _ in range(80):
            enforcer.record_request(latency_ms=3000.0, success=True)
        for _ in range(20):
            enforcer.record_request(latency_ms=3000.0, success=False)

        violations = enforcer.check_violations()
        violation_names = {v.slo_name for v in violations}
        # Should have at least latency and error rate violations
        assert len(violation_names) >= 2


class TestSLOViolationDataclass:
    """Tests for the SLOViolation dataclass."""

    def test_violation_creation(self):
        """SLOViolation is created with correct fields."""
        now = datetime.now(timezone.utc)
        v = SLOViolation(
            slo_name="latency_p95",
            target_value=500.0,
            actual_value=750.0,
            timestamp=now,
            message="p95 latency exceeded",
        )
        assert v.slo_name == "latency_p95"
        assert v.target_value == 500.0
        assert v.actual_value == 750.0

    def test_violation_to_dict(self):
        """SLOViolation serializes to dictionary."""
        now = datetime.now(timezone.utc)
        v = SLOViolation(
            slo_name="error_rate",
            target_value=0.01,
            actual_value=0.05,
            timestamp=now,
            message="Error rate too high",
        )
        d = v.to_dict()
        assert d["slo_name"] == "error_rate"
        assert d["target_value"] == 0.01
        assert d["actual_value"] == 0.05
        assert "timestamp" in d
        assert d["message"] == "Error rate too high"


class TestSLOEnforcerComplianceStatus:
    """Tests for the compliance status endpoint data."""

    def test_compliance_status_structure(self):
        """Compliance status has the expected structure."""
        enforcer = SLOEnforcer()
        for _ in range(100):
            enforcer.record_request(latency_ms=50.0, success=True)

        status = enforcer.get_compliance_status()

        assert "timestamp" in status
        assert "overall_healthy" in status
        assert "window" in status
        assert "slos" in status
        assert "metrics" in status
        assert "violation_count" in status

    def test_compliance_status_healthy(self):
        """Status reports healthy when all SLOs are met."""
        enforcer = SLOEnforcer()
        for _ in range(1000):
            enforcer.record_request(latency_ms=50.0, success=True)

        status = enforcer.get_compliance_status()
        assert status["overall_healthy"] is True
        assert status["violation_count"] == 0

        # Each SLO should be compliant
        for slo_data in status["slos"].values():
            assert slo_data["compliant"] is True

    def test_compliance_status_unhealthy(self):
        """Status reports unhealthy when SLOs are breached."""
        enforcer = SLOEnforcer()
        for _ in range(100):
            enforcer.record_request(latency_ms=600.0, success=True)

        status = enforcer.get_compliance_status()
        assert status["overall_healthy"] is False

    def test_compliance_slo_keys(self):
        """Compliance status includes all four enforced SLOs."""
        enforcer = SLOEnforcer()
        enforcer.record_request(latency_ms=50.0, success=True)

        status = enforcer.get_compliance_status()
        assert "latency_p95" in status["slos"]
        assert "latency_p99" in status["slos"]
        assert "error_rate" in status["slos"]
        assert "availability" in status["slos"]

    def test_compliance_window(self):
        """Compliance status includes window information."""
        enforcer = SLOEnforcer(window_seconds=600.0)
        enforcer.record_request(latency_ms=50.0, success=True)

        status = enforcer.get_compliance_status()
        assert status["window"]["duration_seconds"] == 600.0
        assert "start" in status["window"]
        assert "end" in status["window"]


class TestSLOEnforcerErrorBudget:
    """Tests for the error budget computation."""

    def test_budget_structure(self):
        """Error budget has the expected structure."""
        enforcer = SLOEnforcer()
        for _ in range(100):
            enforcer.record_request(latency_ms=50.0, success=True)

        budget = enforcer.get_error_budget()

        assert "timestamp" in budget
        assert "window" in budget
        assert "budgets" in budget
        assert len(budget["budgets"]) == 4  # availability, error_rate, p95, p99

    def test_budget_all_remaining(self):
        """Full budget remains when all requests are within SLO."""
        enforcer = SLOEnforcer()
        for _ in range(1000):
            enforcer.record_request(latency_ms=50.0, success=True)

        budget = enforcer.get_error_budget()
        for b in budget["budgets"]:
            assert b["error_budget_remaining_pct"] > 0
            assert b["error_budget_consumed_pct"] < 100

    def test_budget_consumed_on_errors(self):
        """Budget is consumed when error rate rises."""
        enforcer = SLOEnforcer()
        # 5% error rate against 1% target = budget consumed
        for _ in range(95):
            enforcer.record_request(latency_ms=50.0, success=True)
        for _ in range(5):
            enforcer.record_request(latency_ms=50.0, success=False)

        budget = enforcer.get_error_budget()
        error_budget = next(
            b for b in budget["budgets"] if b["slo_id"] == "error_rate"
        )
        assert error_budget["error_budget_consumed_pct"] > 0
        assert error_budget["burn_rate"] > 1.0

    def test_budget_consumed_on_high_latency(self):
        """Budget is consumed when latency exceeds target."""
        enforcer = SLOEnforcer()
        for _ in range(100):
            enforcer.record_request(latency_ms=800.0, success=True)

        budget = enforcer.get_error_budget()
        p95_budget = next(
            b for b in budget["budgets"] if b["slo_id"] == "latency_p95"
        )
        assert p95_budget["error_budget_consumed_pct"] > 0

    def test_budget_ids(self):
        """All expected budget IDs are present."""
        enforcer = SLOEnforcer()
        enforcer.record_request(latency_ms=50.0, success=True)

        budget = enforcer.get_error_budget()
        budget_ids = {b["slo_id"] for b in budget["budgets"]}
        assert budget_ids == {"availability", "error_rate", "latency_p95", "latency_p99"}


class TestSLOEnforcerRecentViolations:
    """Tests for recent violations retrieval."""

    def test_no_violations(self):
        """Returns empty list when no violations exist."""
        enforcer = SLOEnforcer()
        assert enforcer.get_recent_violations() == []

    def test_violations_returned(self):
        """Returns violation dictionaries after detection."""
        enforcer = SLOEnforcer()
        for _ in range(100):
            enforcer.record_request(latency_ms=600.0, success=True)
        enforcer.check_violations()

        violations = enforcer.get_recent_violations()
        assert len(violations) > 0
        assert "slo_name" in violations[0]
        assert "target_value" in violations[0]
        assert "actual_value" in violations[0]

    def test_violations_limited(self):
        """Violations are limited to the requested count."""
        enforcer = SLOEnforcer()
        # Create many violations by checking multiple times
        for _ in range(100):
            enforcer.record_request(latency_ms=600.0, success=True)
        for _ in range(10):
            enforcer.check_violations()

        violations = enforcer.get_recent_violations(limit=3)
        assert len(violations) <= 3


class TestSLOEnforcerReset:
    """Tests for enforcer reset."""

    def test_reset_clears_requests(self):
        """Reset clears all tracked requests."""
        enforcer = SLOEnforcer()
        for _ in range(50):
            enforcer.record_request(latency_ms=50.0, success=True)
        enforcer.reset()
        assert len(enforcer._requests) == 0

    def test_reset_clears_violations(self):
        """Reset clears all violations."""
        enforcer = SLOEnforcer()
        for _ in range(100):
            enforcer.record_request(latency_ms=600.0, success=True)
        enforcer.check_violations()

        enforcer.reset()
        assert len(enforcer._violations) == 0
        assert enforcer.get_recent_violations() == []


class TestGlobalSLOEnforcer:
    """Tests for the global enforcer singleton."""

    def setup_method(self):
        """Reset global enforcer before each test."""
        reset_slo_enforcer()

    def teardown_method(self):
        """Reset global enforcer after each test."""
        reset_slo_enforcer()

    def test_get_creates_singleton(self):
        """get_slo_enforcer creates a singleton instance."""
        enforcer1 = get_slo_enforcer()
        enforcer2 = get_slo_enforcer()
        assert enforcer1 is enforcer2

    def test_reset_clears_singleton(self):
        """reset_slo_enforcer clears the singleton."""
        enforcer1 = get_slo_enforcer()
        reset_slo_enforcer()
        enforcer2 = get_slo_enforcer()
        assert enforcer1 is not enforcer2

    def test_global_enforcer_is_slo_enforcer(self):
        """Global enforcer is an SLOEnforcer instance."""
        enforcer = get_slo_enforcer()
        assert isinstance(enforcer, SLOEnforcer)


class TestSLODefaultTargets:
    """Tests for the new SLO default target values."""

    def test_default_p95_latency(self):
        """Default p95 latency target is 500ms."""
        assert DEFAULT_LATENCY_P95_MS == 500

    def test_default_p99_latency(self):
        """Default p99 latency target is 2000ms."""
        assert DEFAULT_LATENCY_P99_MS == 2000

    def test_default_error_rate(self):
        """Default error rate target is 1%."""
        assert DEFAULT_ERROR_RATE_TARGET == 0.01

    def test_default_availability(self):
        """Default availability target is 99.9%."""
        assert DEFAULT_AVAILABILITY_TARGET == 0.999

    def test_targets_include_p95(self):
        """get_slo_targets includes latency_p95."""
        from aragora.observability.slo import get_slo_targets

        targets = get_slo_targets()
        assert "latency_p95" in targets
        assert targets["latency_p95"].target == 0.5  # 500ms in seconds

    def test_targets_include_error_rate(self):
        """get_slo_targets includes error_rate."""
        from aragora.observability.slo import get_slo_targets

        targets = get_slo_targets()
        assert "error_rate" in targets
        assert targets["error_rate"].target == 0.01
        assert targets["error_rate"].comparison == "lte"


class TestSLOEnforcerEdgeCases:
    """Tests for edge cases in the SLO enforcer."""

    def test_single_request(self):
        """Enforcer works with a single request."""
        enforcer = SLOEnforcer()
        enforcer.record_request(latency_ms=50.0, success=True)

        metrics = enforcer.get_metrics()
        assert metrics["total_requests"] == 1
        assert metrics["latency_p50_ms"] == 50.0
        assert metrics["latency_p95_ms"] == 50.0
        assert metrics["latency_p99_ms"] == 50.0

    def test_all_failures(self):
        """Enforcer handles 100% failure rate."""
        enforcer = SLOEnforcer()
        for _ in range(100):
            enforcer.record_request(latency_ms=100.0, success=False)

        metrics = enforcer.get_metrics()
        assert metrics["error_rate"] == 1.0
        assert metrics["uptime"] == 0.0

        violations = enforcer.check_violations()
        violation_names = {v.slo_name for v in violations}
        assert "error_rate" in violation_names
        assert "availability" in violation_names

    def test_exactly_at_threshold(self):
        """Requests exactly at the threshold are compliant."""
        enforcer = SLOEnforcer()
        # Exactly 500ms p95 target
        for _ in range(100):
            enforcer.record_request(latency_ms=500.0, success=True)

        violations = enforcer.check_violations()
        violation_names = [v.slo_name for v in violations]
        assert "latency_p95" not in violation_names

    def test_error_rate_exactly_at_1_percent(self):
        """Error rate exactly at 1% is still compliant."""
        enforcer = SLOEnforcer()
        for _ in range(99):
            enforcer.record_request(latency_ms=50.0, success=True)
        enforcer.record_request(latency_ms=50.0, success=False)

        violations = enforcer.check_violations()
        violation_names = [v.slo_name for v in violations]
        assert "error_rate" not in violation_names
