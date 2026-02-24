"""
Tests for load testing harness and SLO enforcement.

Tests LoadTestRunner, LoadTestConfig, LoadTestResult, SLOEnforcer,
and related dataclasses.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.observability.load_test import (
    LoadTestConfig,
    LoadTestResult,
    LoadTestRunner,
    SLOCheckResult,
    SLOEnforcer,
    SLOTargetDef,
    SLOViolation,
    _parse_period,
    _percentile,
)


# =============================================================================
# LoadTestConfig Tests
# =============================================================================


class TestLoadTestConfig:
    """Tests for LoadTestConfig dataclass."""

    def test_default_values(self):
        """Default config has sensible values."""
        config = LoadTestConfig()
        assert config.target_rps == 100
        assert config.duration_seconds == 30
        assert config.concurrency == 10
        assert config.warmup_seconds == 2.0

    def test_custom_values(self):
        """Config accepts custom values."""
        config = LoadTestConfig(
            target_rps=500,
            duration_seconds=60,
            concurrency=50,
            warmup_seconds=5.0,
        )
        assert config.target_rps == 500
        assert config.duration_seconds == 60
        assert config.concurrency == 50
        assert config.warmup_seconds == 5.0


# =============================================================================
# LoadTestResult Tests
# =============================================================================


class TestLoadTestResult:
    """Tests for LoadTestResult dataclass."""

    def test_default_values(self):
        """Default result has zero values."""
        result = LoadTestResult()
        assert result.total_requests == 0
        assert result.successful == 0
        assert result.failed == 0
        assert result.p50_ms == 0.0
        assert result.p95_ms == 0.0
        assert result.p99_ms == 0.0
        assert result.max_ms == 0.0
        assert result.rps_achieved == 0.0
        assert result.errors_by_type == {}

    def test_populated_result(self):
        """Result stores populated values correctly."""
        result = LoadTestResult(
            total_requests=1000,
            successful=990,
            failed=10,
            p50_ms=15.0,
            p95_ms=45.0,
            p99_ms=120.0,
            max_ms=500.0,
            rps_achieved=98.5,
            errors_by_type={"ConnectionError": 7, "TimeoutError": 3},
        )
        assert result.total_requests == 1000
        assert result.successful == 990
        assert result.failed == 10
        assert result.p95_ms == 45.0
        assert result.errors_by_type["ConnectionError"] == 7


# =============================================================================
# Percentile Helper Tests
# =============================================================================


class TestPercentile:
    """Tests for the _percentile helper."""

    def test_empty_list(self):
        """Empty list returns 0.0."""
        assert _percentile([], 50) == 0.0

    def test_single_value(self):
        """Single value list returns that value for any percentile."""
        assert _percentile([42.0], 50) == 42.0
        assert _percentile([42.0], 99) == 42.0

    def test_p50(self):
        """P50 returns the median-area value."""
        values = sorted([10.0, 20.0, 30.0, 40.0, 50.0])
        p50 = _percentile(values, 50)
        assert 20.0 <= p50 <= 30.0

    def test_p99_large_list(self):
        """P99 on larger list returns near-max value."""
        values = sorted(float(i) for i in range(100))
        p99 = _percentile(values, 99)
        assert p99 >= 95.0


# =============================================================================
# LoadTestRunner Tests
# =============================================================================


class TestLoadTestRunner:
    """Tests for LoadTestRunner."""

    @pytest.mark.asyncio
    async def test_run_with_custom_request_func(self):
        """Runner works with a custom request function."""
        call_count = 0

        async def mock_request() -> bool:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.001)
            return True

        runner = LoadTestRunner()
        config = LoadTestConfig(
            target_rps=200,
            duration_seconds=1,
            concurrency=10,
            warmup_seconds=0.0,
        )

        result = await runner.run("http://test", config, request_func=mock_request)

        assert result.total_requests > 0
        assert result.successful > 0
        assert result.failed == 0
        assert result.p50_ms >= 0
        assert result.p95_ms >= 0
        assert result.rps_achieved > 0

    @pytest.mark.asyncio
    async def test_run_with_failing_requests(self):
        """Runner tracks failed requests."""

        async def failing_request() -> bool:
            await asyncio.sleep(0.001)
            return False

        runner = LoadTestRunner()
        config = LoadTestConfig(
            target_rps=100,
            duration_seconds=1,
            concurrency=5,
            warmup_seconds=0.0,
        )

        result = await runner.run("http://test", config, request_func=failing_request)

        assert result.total_requests > 0
        assert result.successful == 0
        assert result.failed > 0

    @pytest.mark.asyncio
    async def test_run_with_mixed_results(self):
        """Runner correctly counts mixed success/failure."""
        counter = 0

        async def mixed_request() -> bool:
            nonlocal counter
            counter += 1
            await asyncio.sleep(0.001)
            return counter % 3 != 0  # Fail every 3rd request

        runner = LoadTestRunner()
        config = LoadTestConfig(
            target_rps=100,
            duration_seconds=1,
            concurrency=5,
            warmup_seconds=0.0,
        )

        result = await runner.run("http://test", config, request_func=mixed_request)

        assert result.total_requests > 0
        assert result.successful > 0
        assert result.failed > 0
        assert result.successful + result.failed == result.total_requests

    @pytest.mark.asyncio
    async def test_run_with_http_client(self):
        """Runner uses provided http_client."""
        mock_response = MagicMock(spec=[])
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        runner = LoadTestRunner(http_client=mock_client)
        config = LoadTestConfig(
            target_rps=50,
            duration_seconds=1,
            concurrency=5,
            warmup_seconds=0.0,
        )

        result = await runner.run("http://test/health", config)

        assert result.total_requests > 0
        assert result.successful > 0
        mock_client.get.assert_called()

    @pytest.mark.asyncio
    async def test_run_debate_load_test(self):
        """Debate load test uses POST requests."""
        mock_response = MagicMock(spec=[])
        mock_response.status_code = 201
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        runner = LoadTestRunner(http_client=mock_client)
        config = LoadTestConfig(
            target_rps=50,
            duration_seconds=1,
            concurrency=5,
            warmup_seconds=0.0,
        )

        result = await runner.run_debate_load_test(config=config)

        assert result.total_requests > 0
        assert result.successful > 0
        mock_client.post.assert_called()

    @pytest.mark.asyncio
    async def test_warmup_not_counted(self):
        """Warmup requests are not counted in results."""
        total_calls = 0

        async def counting_request() -> bool:
            nonlocal total_calls
            total_calls += 1
            await asyncio.sleep(0.001)
            return True

        runner = LoadTestRunner()
        config = LoadTestConfig(
            target_rps=100,
            duration_seconds=1,
            concurrency=5,
            warmup_seconds=0.5,
        )

        result = await runner.run("http://test", config, request_func=counting_request)

        # Total calls includes warmup, but result only counts measurement phase
        assert total_calls > result.total_requests

    @pytest.mark.asyncio
    async def test_concurrency_limited(self):
        """Semaphore limits concurrent requests."""
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def tracking_request() -> bool:
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent
            await asyncio.sleep(0.01)
            async with lock:
                current_concurrent -= 1
            return True

        runner = LoadTestRunner()
        config = LoadTestConfig(
            target_rps=500,
            duration_seconds=1,
            concurrency=3,
            warmup_seconds=0.0,
        )

        await runner.run("http://test", config, request_func=tracking_request)

        assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_http_client_error_handling(self):
        """Runner handles http_client exceptions gracefully."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=ConnectionError("refused"))

        runner = LoadTestRunner(http_client=mock_client)
        config = LoadTestConfig(
            target_rps=50,
            duration_seconds=1,
            concurrency=5,
            warmup_seconds=0.0,
        )

        result = await runner.run("http://test/health", config)

        # Requests should still complete (as failures)
        assert result.total_requests > 0
        assert result.failed > 0


# =============================================================================
# SLOTargetDef Tests
# =============================================================================


class TestSLOTargetDef:
    """Tests for SLOTargetDef dataclass."""

    def test_creation(self):
        """SLO target is created with all fields."""
        target = SLOTargetDef(
            name="api_latency",
            metric="api_p95",
            threshold=200.0,
            window_seconds=3600,
        )
        assert target.name == "api_latency"
        assert target.metric == "api_p95"
        assert target.threshold == 200.0
        assert target.window_seconds == 3600

    def test_default_window(self):
        """Default window is 3600 seconds."""
        target = SLOTargetDef(name="test", metric="m", threshold=1.0)
        assert target.window_seconds == 3600


# =============================================================================
# SLOCheckResult Tests
# =============================================================================


class TestSLOCheckResult:
    """Tests for SLOCheckResult dataclass."""

    def test_compliant_result(self):
        """Compliant check result has correct fields."""
        result = SLOCheckResult(
            target_name="api_p95",
            metric="api_p95",
            threshold=200.0,
            current_value=150.0,
            compliant=True,
        )
        assert result.compliant is True
        assert result.current_value == 150.0
        assert isinstance(result.checked_at, datetime)

    def test_non_compliant_result(self):
        """Non-compliant check result has correct fields."""
        result = SLOCheckResult(
            target_name="api_p95",
            metric="api_p95",
            threshold=200.0,
            current_value=350.0,
            compliant=False,
        )
        assert result.compliant is False
        assert result.current_value == 350.0


# =============================================================================
# SLOViolation Tests
# =============================================================================


class TestSLOViolation:
    """Tests for SLOViolation dataclass."""

    def test_violation_creation(self):
        """Violation is created with all fields."""
        violation = SLOViolation(
            target_name="api_latency",
            metric="api_p95",
            threshold=200.0,
            observed_value=350.0,
            message="API latency exceeded threshold",
        )
        assert violation.target_name == "api_latency"
        assert violation.observed_value == 350.0
        assert isinstance(violation.occurred_at, datetime)


# =============================================================================
# SLOEnforcer Tests
# =============================================================================


class TestSLOEnforcer:
    """Tests for SLOEnforcer."""

    def test_default_targets(self):
        """Enforcer starts with default SLO targets."""
        enforcer = SLOEnforcer()
        targets = enforcer.get_targets()
        assert len(targets) == 3

        names = {t.name for t in targets}
        assert "debate_p95_latency" in names
        assert "api_p95_latency" in names
        assert "error_rate" in names

    def test_register_target(self):
        """Registering a new target adds it."""
        enforcer = SLOEnforcer()
        custom = SLOTargetDef(
            name="custom_metric",
            metric="custom",
            threshold=42.0,
        )
        enforcer.register_target(custom)

        targets = enforcer.get_targets()
        assert any(t.name == "custom_metric" for t in targets)

    def test_register_target_replaces_existing(self):
        """Registering a target with same name replaces it."""
        enforcer = SLOEnforcer()
        original_count = len(enforcer.get_targets())

        replacement = SLOTargetDef(
            name="api_p95_latency",
            metric="api_p95",
            threshold=500.0,  # Relaxed threshold
        )
        enforcer.register_target(replacement)

        targets = enforcer.get_targets()
        assert len(targets) == original_count
        api_target = next(t for t in targets if t.name == "api_p95_latency")
        assert api_target.threshold == 500.0

    def test_record_metric_no_violation(self):
        """Recording a metric within threshold produces no violations."""
        enforcer = SLOEnforcer()
        violations = enforcer.record_metric("api_p95", 150.0)
        assert violations == []

    def test_record_metric_with_violation(self):
        """Recording a metric above threshold triggers violation."""
        enforcer = SLOEnforcer()
        violations = enforcer.record_metric("api_p95", 300.0)
        assert len(violations) == 1
        assert violations[0].target_name == "api_p95_latency"
        assert violations[0].observed_value == 300.0

    def test_record_error_rate_violation(self):
        """Recording error rate above threshold triggers violation."""
        enforcer = SLOEnforcer()
        violations = enforcer.record_metric("error_rate", 0.05)
        assert len(violations) == 1
        assert violations[0].target_name == "error_rate"

    def test_record_error_rate_compliant(self):
        """Recording error rate within threshold produces no violations."""
        enforcer = SLOEnforcer()
        violations = enforcer.record_metric("error_rate", 0.005)
        assert violations == []

    def test_check_slo_no_data(self):
        """Checking SLO with no data returns compliant."""
        enforcer = SLOEnforcer()
        target = enforcer.get_targets()[0]
        result = enforcer.check_slo(target)

        assert result.compliant is True
        assert result.current_value == 0.0

    def test_check_slo_compliant(self):
        """Checking SLO with compliant data returns compliant."""
        enforcer = SLOEnforcer()
        enforcer.record_metric("api_p95", 100.0)

        target = next(t for t in enforcer.get_targets() if t.metric == "api_p95")
        result = enforcer.check_slo(target)

        assert result.compliant is True
        assert result.current_value == 100.0

    def test_check_slo_non_compliant(self):
        """Checking SLO with violating data returns non-compliant."""
        enforcer = SLOEnforcer()
        enforcer.record_metric("api_p95", 500.0)

        target = next(t for t in enforcer.get_targets() if t.metric == "api_p95")
        result = enforcer.check_slo(target)

        assert result.compliant is False
        assert result.current_value == 500.0

    def test_check_all(self):
        """check_all returns results for all targets."""
        enforcer = SLOEnforcer()
        results = enforcer.check_all()
        assert len(results) == 3

    def test_get_violations_empty(self):
        """get_violations returns empty list when no violations."""
        enforcer = SLOEnforcer()
        assert enforcer.get_violations() == []

    def test_get_violations_within_period(self):
        """get_violations returns violations within the period."""
        enforcer = SLOEnforcer()
        enforcer.record_metric("api_p95", 500.0)
        enforcer.record_metric("debate_p95", 10000.0)

        violations = enforcer.get_violations(period="1h")
        assert len(violations) == 2

    def test_get_violations_period_filter(self):
        """get_violations filters out old violations."""
        enforcer = SLOEnforcer()
        enforcer.record_metric("api_p95", 500.0)

        # Manually age a violation
        enforcer._violations[0].occurred_at = datetime.now(timezone.utc) - timedelta(hours=48)

        violations = enforcer.get_violations(period="24h")
        assert len(violations) == 0

    def test_get_violations_days_period(self):
        """get_violations supports day period format."""
        enforcer = SLOEnforcer()
        enforcer.record_metric("api_p95", 500.0)

        violations = enforcer.get_violations(period="7d")
        assert len(violations) == 1

    def test_clear_violations(self):
        """clear_violations removes all violations."""
        enforcer = SLOEnforcer()
        enforcer.record_metric("api_p95", 500.0)
        assert len(enforcer.get_violations()) == 1

        enforcer.clear_violations()
        assert len(enforcer.get_violations()) == 0

    def test_prune_observations(self):
        """prune_observations removes old observations."""
        enforcer = SLOEnforcer()
        enforcer.record_metric("api_p95", 100.0)

        # Age the observation
        enforcer._observations["api_p95"][0] = (
            datetime.now(timezone.utc) - timedelta(hours=3),
            100.0,
        )

        pruned = enforcer.prune_observations(max_age_seconds=3600)
        assert pruned == 1
        assert len(enforcer._observations["api_p95"]) == 0

    def test_unregistered_metric_ignored(self):
        """Recording a metric with no matching target produces no violations."""
        enforcer = SLOEnforcer()
        violations = enforcer.record_metric("unknown_metric", 9999.0)
        assert violations == []


# =============================================================================
# Period Parser Tests
# =============================================================================


class TestParsePeriod:
    """Tests for _parse_period helper."""

    def test_hours(self):
        """Parse hours format."""
        delta = _parse_period("24h")
        assert delta == timedelta(hours=24)

    def test_days(self):
        """Parse days format."""
        delta = _parse_period("7d")
        assert delta == timedelta(days=7)

    def test_case_insensitive(self):
        """Parse is case insensitive."""
        delta = _parse_period("12H")
        assert delta == timedelta(hours=12)

    def test_invalid_format(self):
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported period format"):
            _parse_period("5m")

    def test_whitespace_stripped(self):
        """Whitespace is stripped."""
        delta = _parse_period("  1h  ")
        assert delta == timedelta(hours=1)
