"""
Tests for the streaming SLO baseline load test infrastructure.

Validates SLO threshold evaluation, concurrent debate simulation,
latency measurement accuracy, report generation, and graceful
degradation under load.

Run with: pytest tests/performance/test_slo_baseline.py -v
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.load_test_baseline import (
    BaselineResult,
    DebateMetrics,
    MockStreamingAgent,
    SLOCheckResult,
    StreamingSLO,
    evaluate_all_slos,
    evaluate_slo,
    format_report,
    get_streaming_slo_targets,
    percentile,
    run_baseline,
    run_single_debate,
    simulate_reconnection,
)


# =============================================================================
# SLO Definition Tests
# =============================================================================


class TestStreamingSLOTargets:
    """Tests for streaming SLO target definitions."""

    def test_four_slos_defined(self):
        """Exactly four streaming SLOs must be defined."""
        slos = get_streaming_slo_targets()
        assert len(slos) == 4

    def test_slo_ids(self):
        """SLO ids match the expected set."""
        slos = get_streaming_slo_targets()
        expected = {
            "first_byte_latency",
            "message_throughput",
            "reconnection_success_rate",
            "debate_completion",
        }
        assert set(slos.keys()) == expected

    def test_first_byte_latency_defaults(self):
        """First byte latency has correct default target."""
        slos = get_streaming_slo_targets()
        slo = slos["first_byte_latency"]
        assert slo.target == 500.0
        assert slo.comparison == "lte"
        assert slo.unit == "ms"

    def test_message_throughput_defaults(self):
        """Message throughput has correct default target."""
        slos = get_streaming_slo_targets()
        slo = slos["message_throughput"]
        assert slo.target == 10.0
        assert slo.comparison == "gte"
        assert slo.unit == "messages/sec/debate"

    def test_reconnection_rate_defaults(self):
        """Reconnection success rate has correct default target."""
        slos = get_streaming_slo_targets()
        slo = slos["reconnection_success_rate"]
        assert slo.target == 0.99
        assert slo.comparison == "gte"
        assert slo.unit == "ratio"

    def test_debate_completion_defaults(self):
        """Debate completion has correct default target."""
        slos = get_streaming_slo_targets()
        slo = slos["debate_completion"]
        assert slo.target == 30.0
        assert slo.comparison == "lte"
        assert slo.unit == "seconds"

    def test_environment_override(self, monkeypatch):
        """SLO targets can be overridden via environment variables."""
        monkeypatch.setenv("SLO_FIRST_BYTE_P95_MS", "250")
        monkeypatch.setenv("SLO_MSG_THROUGHPUT_MIN", "20")
        monkeypatch.setenv("SLO_RECONNECT_RATE_MIN", "0.999")
        monkeypatch.setenv("SLO_COMPLETION_P99_S", "15")

        slos = get_streaming_slo_targets()
        assert slos["first_byte_latency"].target == 250.0
        assert slos["message_throughput"].target == 20.0
        assert slos["reconnection_success_rate"].target == 0.999
        assert slos["debate_completion"].target == 15.0


# =============================================================================
# Percentile Calculation Tests
# =============================================================================


class TestPercentile:
    """Tests for the percentile calculation utility."""

    def test_empty_list(self):
        """Empty list returns 0.0."""
        assert percentile([], 50) == 0.0

    def test_single_value(self):
        """Single value returns that value for any percentile."""
        assert percentile([42.0], 50) == 42.0
        assert percentile([42.0], 99) == 42.0

    def test_p50_median(self):
        """P50 returns the median area value."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        p50 = percentile(values, 50)
        assert 25.0 <= p50 <= 35.0

    def test_p95_near_max(self):
        """P95 on a 100-element list returns near-max value."""
        values = [float(i) for i in range(100)]
        p95 = percentile(values, 95)
        assert p95 >= 93.0

    def test_p99_on_large_list(self):
        """P99 on a large list returns the tail value."""
        values = [float(i) for i in range(1000)]
        p99 = percentile(values, 99)
        assert p99 >= 985.0

    def test_interpolation(self):
        """Percentile uses linear interpolation between adjacent values."""
        values = [0.0, 100.0]
        p50 = percentile(values, 50)
        assert p50 == pytest.approx(50.0, abs=0.01)


# =============================================================================
# SLO Evaluation Tests
# =============================================================================


class TestSLOEvaluation:
    """Tests for SLO threshold evaluation logic."""

    def test_lte_passing(self):
        """SLO with lte comparison passes when value is below target."""
        slo = StreamingSLO(
            slo_id="test",
            name="Test",
            target=500.0,
            unit="ms",
            comparison="lte",
            description="Test SLO",
        )
        result = evaluate_slo(slo, 300.0)
        assert result.passed is True
        assert result.margin == pytest.approx(200.0)

    def test_lte_failing(self):
        """SLO with lte comparison fails when value exceeds target."""
        slo = StreamingSLO(
            slo_id="test",
            name="Test",
            target=500.0,
            unit="ms",
            comparison="lte",
            description="Test SLO",
        )
        result = evaluate_slo(slo, 600.0)
        assert result.passed is False
        assert result.margin == pytest.approx(-100.0)

    def test_lte_boundary(self):
        """SLO with lte comparison passes at exact boundary."""
        slo = StreamingSLO(
            slo_id="test",
            name="Test",
            target=500.0,
            unit="ms",
            comparison="lte",
            description="Test SLO",
        )
        result = evaluate_slo(slo, 500.0)
        assert result.passed is True
        assert result.margin == pytest.approx(0.0)

    def test_gte_passing(self):
        """SLO with gte comparison passes when value meets target."""
        slo = StreamingSLO(
            slo_id="test",
            name="Test",
            target=10.0,
            unit="msg/s",
            comparison="gte",
            description="Test SLO",
        )
        result = evaluate_slo(slo, 15.0)
        assert result.passed is True
        assert result.margin == pytest.approx(5.0)

    def test_gte_failing(self):
        """SLO with gte comparison fails when value is below target."""
        slo = StreamingSLO(
            slo_id="test",
            name="Test",
            target=10.0,
            unit="msg/s",
            comparison="gte",
            description="Test SLO",
        )
        result = evaluate_slo(slo, 8.0)
        assert result.passed is False
        assert result.margin == pytest.approx(-2.0)

    def test_result_to_dict(self):
        """SLOCheckResult.to_dict produces correct keys."""
        result = SLOCheckResult(
            slo_id="test",
            name="Test SLO",
            target=500.0,
            actual=300.0,
            unit="ms",
            passed=True,
            description="A test SLO",
            margin=200.0,
        )
        d = result.to_dict()
        assert d["slo_id"] == "test"
        assert d["passed"] is True
        assert d["margin"] == 200.0
        assert "description" in d


# =============================================================================
# Mock Agent Tests
# =============================================================================


class TestMockStreamingAgent:
    """Tests for the mock streaming agent."""

    @pytest.mark.asyncio
    async def test_stream_proposal_returns_timestamps(self):
        """stream_proposal returns a list of monotonic timestamps."""
        agent = MockStreamingAgent("test_agent", latency_range=(0.001, 0.002))
        timestamps = await agent.stream_proposal("test prompt", num_tokens=5)
        assert len(timestamps) == 5
        # Timestamps should be monotonically increasing
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    @pytest.mark.asyncio
    async def test_critique_returns_string(self):
        """critique returns a string response."""
        agent = MockStreamingAgent("test_agent", latency_range=(0.001, 0.002))
        result = await agent.critique("test proposal")
        assert isinstance(result, str)
        assert "test_agent" in result

    @pytest.mark.asyncio
    async def test_vote_returns_valid_index(self):
        """vote returns a valid proposal index."""
        agent = MockStreamingAgent("test_agent", latency_range=(0.001, 0.002))
        result = await agent.vote(["proposal_a", "proposal_b", "proposal_c"])
        assert isinstance(result, int)
        assert 0 <= result <= 2


# =============================================================================
# Reconnection Simulation Tests
# =============================================================================


class TestReconnectionSimulation:
    """Tests for the WebSocket reconnection simulator."""

    @pytest.mark.asyncio
    async def test_reconnection_mostly_succeeds(self):
        """Reconnection attempts succeed at a high rate."""
        results = [await simulate_reconnection(fail_probability=0.01) for _ in range(100)]
        success_rate = sum(results) / len(results)
        # With 1% failure, expect > 90% success in a sample of 100
        assert success_rate > 0.90

    @pytest.mark.asyncio
    async def test_reconnection_zero_fail_always_succeeds(self):
        """Zero failure probability means 100% success."""
        results = [await simulate_reconnection(fail_probability=0.0) for _ in range(50)]
        assert all(results)

    @pytest.mark.asyncio
    async def test_reconnection_total_fail(self):
        """Total failure probability means 0% success."""
        results = [await simulate_reconnection(fail_probability=1.0) for _ in range(50)]
        assert not any(results)


# =============================================================================
# Single Debate Simulation Tests
# =============================================================================


class TestSingleDebateSimulation:
    """Tests for individual debate simulation."""

    @pytest.mark.asyncio
    async def test_completed_debate_has_metrics(self):
        """A completed debate produces all expected metrics."""
        metrics = await run_single_debate(
            debate_id="test_001",
            num_agents=2,
            num_rounds=2,
            tokens_per_proposal=5,
            reconnect_probability=0.0,
            fail_rate=0.0,
        )
        assert metrics.completed is True
        assert metrics.first_byte_ms > 0
        assert metrics.total_messages > 0
        assert metrics.duration_s > 0
        assert metrics.messages_per_sec > 0
        assert metrics.error is None

    @pytest.mark.asyncio
    async def test_first_byte_latency_is_positive(self):
        """First byte latency is a positive value in milliseconds."""
        metrics = await run_single_debate(
            debate_id="test_002",
            num_agents=2,
            num_rounds=1,
            tokens_per_proposal=3,
            fail_rate=0.0,
        )
        assert metrics.first_byte_ms > 0
        # With mock agents (5-30ms per token), first byte should be < 200ms
        assert metrics.first_byte_ms < 200

    @pytest.mark.asyncio
    async def test_message_count_scales_with_rounds(self):
        """More rounds produce more messages."""
        metrics_1 = await run_single_debate(
            debate_id="test_1r",
            num_agents=2,
            num_rounds=1,
            fail_rate=0.0,
        )
        metrics_3 = await run_single_debate(
            debate_id="test_3r",
            num_agents=2,
            num_rounds=3,
            fail_rate=0.0,
        )
        assert metrics_3.total_messages > metrics_1.total_messages

    @pytest.mark.asyncio
    async def test_forced_failure(self):
        """A debate with 100% failure rate produces an error."""
        metrics = await run_single_debate(
            debate_id="test_fail",
            num_agents=2,
            num_rounds=1,
            fail_rate=1.0,
        )
        assert metrics.completed is False
        assert metrics.error is not None


# =============================================================================
# SLO Evaluation Against Metrics Tests
# =============================================================================


class TestEvaluateAllSLOs:
    """Tests for evaluating all four SLOs against debate metrics."""

    def _make_metrics(
        self,
        count: int = 10,
        first_byte_ms: float = 100.0,
        msgs_per_sec: float = 50.0,
        duration_s: float = 2.0,
        reconnect_attempts: int = 1,
        reconnect_successes: int = 1,
        completed: bool = True,
    ) -> list[DebateMetrics]:
        """Create a list of DebateMetrics with uniform values."""
        return [
            DebateMetrics(
                debate_id=f"test_{i:03d}",
                first_byte_ms=first_byte_ms + i * 0.1,
                total_messages=int(msgs_per_sec * duration_s),
                duration_s=duration_s,
                messages_per_sec=msgs_per_sec,
                reconnect_attempts=reconnect_attempts,
                reconnect_successes=reconnect_successes,
                completed=completed,
            )
            for i in range(count)
        ]

    def test_all_slos_pass_with_good_metrics(self):
        """All SLOs pass with healthy metric values."""
        metrics = self._make_metrics(
            count=100,
            first_byte_ms=100.0,
            msgs_per_sec=50.0,
            duration_s=2.0,
        )
        results = evaluate_all_slos(metrics)
        assert all(r.passed for r in results.values()), (
            f"Failed SLOs: {[r.name for r in results.values() if not r.passed]}"
        )

    def test_first_byte_latency_slo_fails(self):
        """First byte latency SLO fails when p95 exceeds 500ms."""
        metrics = self._make_metrics(count=100, first_byte_ms=600.0)
        results = evaluate_all_slos(metrics)
        assert results["first_byte_latency"].passed is False

    def test_message_throughput_slo_fails(self):
        """Message throughput SLO fails when p5 is below 10 msg/s."""
        metrics = self._make_metrics(count=100, msgs_per_sec=5.0)
        results = evaluate_all_slos(metrics)
        assert results["message_throughput"].passed is False

    def test_reconnection_rate_slo_fails(self):
        """Reconnection SLO fails when success rate drops below 99%."""
        metrics = self._make_metrics(
            count=100,
            reconnect_attempts=100,
            reconnect_successes=90,
        )
        results = evaluate_all_slos(metrics)
        assert results["reconnection_success_rate"].passed is False

    def test_debate_completion_slo_fails(self):
        """Debate completion SLO fails when p99 exceeds 30s."""
        metrics = self._make_metrics(count=100, duration_s=35.0)
        results = evaluate_all_slos(metrics)
        assert results["debate_completion"].passed is False

    def test_no_reconnect_attempts_passes(self):
        """With zero reconnection attempts, the SLO passes (1/1 = 100%)."""
        metrics = self._make_metrics(
            count=10,
            reconnect_attempts=0,
            reconnect_successes=0,
        )
        results = evaluate_all_slos(metrics)
        assert results["reconnection_success_rate"].passed is True

    def test_empty_metrics_graceful(self):
        """evaluate_all_slos handles empty metrics without crashing."""
        results = evaluate_all_slos([])
        # With no data, percentile returns 0 which passes lte targets
        assert len(results) == 4


# =============================================================================
# Report Generation Tests
# =============================================================================


class TestReportGeneration:
    """Tests for report formatting and JSON output."""

    def _make_baseline_result(self) -> BaselineResult:
        """Create a minimal BaselineResult for testing."""
        metrics = [
            DebateMetrics(
                debate_id=f"test_{i:03d}",
                first_byte_ms=100.0 + i,
                total_messages=50,
                duration_s=2.0,
                messages_per_sec=25.0,
                reconnect_attempts=1,
                reconnect_successes=1,
                completed=True,
            )
            for i in range(10)
        ]
        result = BaselineResult(
            concurrency=10,
            duration_seconds=30.0,
            total_debates=10,
            debate_metrics=metrics,
            actual_duration_s=30.5,
            started_at="2026-02-24T00:00:00+00:00",
            completed_at="2026-02-24T00:00:30+00:00",
        )
        result.slo_results = evaluate_all_slos(metrics)
        result.all_slos_passed = all(r.passed for r in result.slo_results.values())
        return result

    def test_to_dict_contains_required_keys(self):
        """to_dict output has all required top-level keys."""
        result = self._make_baseline_result()
        d = result.to_dict()
        assert "configuration" in d
        assert "timing" in d
        assert "results" in d
        assert "metrics" in d
        assert "slo_validation" in d
        assert "all_slos_passed" in d
        assert "errors" in d

    def test_to_dict_is_json_serializable(self):
        """to_dict output can be serialized to JSON without errors."""
        result = self._make_baseline_result()
        d = result.to_dict()
        json_str = json.dumps(d, indent=2)
        assert len(json_str) > 100

    def test_metrics_section_structure(self):
        """Metrics section has expected subsections."""
        result = self._make_baseline_result()
        d = result.to_dict()
        metrics = d["metrics"]
        assert "first_byte_latency_ms" in metrics
        assert "message_throughput_per_sec" in metrics
        assert "reconnection" in metrics
        assert "debate_completion_s" in metrics

    def test_format_report_contains_slo_status(self):
        """Human-readable report contains SLO pass/fail status."""
        result = self._make_baseline_result()
        report = format_report(result)
        assert "SLO Validation:" in report
        assert "PASS" in report or "FAIL" in report

    def test_format_report_contains_overall(self):
        """Report contains an overall pass/fail summary."""
        result = self._make_baseline_result()
        report = format_report(result)
        assert "Overall:" in report

    def test_failed_debates_in_errors(self):
        """Failed debates appear in the errors section of the report."""
        metrics = [
            DebateMetrics(
                debate_id="fail_001",
                completed=False,
                error="Test failure",
            ),
        ]
        result = BaselineResult(
            debate_metrics=metrics,
            slo_results=evaluate_all_slos(metrics),
        )
        d = result.to_dict()
        assert "Test failure" in d["errors"]


# =============================================================================
# Concurrent Debate Simulation Tests
# =============================================================================


class TestConcurrentSimulation:
    """Tests for concurrent debate simulation accuracy."""

    @pytest.mark.asyncio
    async def test_concurrent_debates_complete(self):
        """Multiple concurrent debates all complete successfully."""
        tasks = [
            run_single_debate(
                debate_id=f"concurrent_{i}",
                num_agents=2,
                num_rounds=1,
                tokens_per_proposal=3,
                fail_rate=0.0,
            )
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)
        completed = [m for m in results if m.completed]
        assert len(completed) == 10

    @pytest.mark.asyncio
    async def test_concurrent_debates_have_distinct_ids(self):
        """Each concurrent debate retains its unique ID."""
        tasks = [
            run_single_debate(
                debate_id=f"id_{i:04d}",
                num_agents=2,
                num_rounds=1,
                tokens_per_proposal=3,
                fail_rate=0.0,
            )
            for i in range(20)
        ]
        results = await asyncio.gather(*tasks)
        ids = [m.debate_id for m in results]
        assert len(set(ids)) == 20


# =============================================================================
# Latency Measurement Accuracy Tests
# =============================================================================


class TestLatencyMeasurement:
    """Tests for latency measurement accuracy."""

    @pytest.mark.asyncio
    async def test_first_byte_within_agent_latency_range(self):
        """First byte latency is bounded by the agent's latency range."""
        # Agent latency range is 5-30ms per token; first byte = first token
        metrics = await run_single_debate(
            debate_id="latency_test",
            num_agents=1,
            num_rounds=1,
            tokens_per_proposal=5,
            fail_rate=0.0,
        )
        # First byte should be roughly in the agent latency range
        # Allow some overhead for task scheduling
        assert metrics.first_byte_ms < 150  # generous upper bound

    @pytest.mark.asyncio
    async def test_duration_scales_with_rounds(self):
        """Debate duration increases with more rounds."""
        metrics_1r = await run_single_debate(
            debate_id="dur_1r",
            num_agents=2,
            num_rounds=1,
            tokens_per_proposal=3,
            fail_rate=0.0,
        )
        metrics_5r = await run_single_debate(
            debate_id="dur_5r",
            num_agents=2,
            num_rounds=5,
            tokens_per_proposal=3,
            fail_rate=0.0,
        )
        assert metrics_5r.duration_s > metrics_1r.duration_s


# =============================================================================
# Graceful Degradation Tests
# =============================================================================


class TestGracefulDegradation:
    """Tests for graceful behavior under adverse conditions."""

    @pytest.mark.asyncio
    async def test_partial_failure_still_produces_results(self):
        """Even with failures, completed debates produce valid metrics."""
        tasks = [
            run_single_debate(
                debate_id=f"degrade_{i}",
                num_agents=2,
                num_rounds=1,
                tokens_per_proposal=3,
                fail_rate=0.3,  # 30% failure rate
            )
            for i in range(50)
        ]
        results = await asyncio.gather(*tasks)
        completed = [m for m in results if m.completed]
        failed = [m for m in results if not m.completed]

        # At least some should complete, some should fail
        assert len(completed) > 0
        assert len(failed) > 0

        # Completed ones should have valid metrics
        for m in completed:
            assert m.first_byte_ms > 0
            assert m.total_messages > 0

    @pytest.mark.asyncio
    async def test_all_failures_handled_gracefully(self):
        """100% failure rate produces metrics with all errors."""
        tasks = [
            run_single_debate(
                debate_id=f"allfail_{i}",
                num_agents=2,
                num_rounds=1,
                fail_rate=1.0,
            )
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)
        assert all(not m.completed for m in results)
        assert all(m.error is not None for m in results)

    def test_evaluate_slos_with_mixed_results(self):
        """SLO evaluation handles a mix of completed and failed debates."""
        metrics = [
            DebateMetrics(
                debate_id=f"ok_{i}",
                first_byte_ms=100.0,
                total_messages=50,
                duration_s=2.0,
                messages_per_sec=25.0,
                reconnect_attempts=1,
                reconnect_successes=1,
                completed=True,
            )
            for i in range(8)
        ] + [
            DebateMetrics(
                debate_id=f"fail_{i}",
                completed=False,
                error="Test failure",
            )
            for i in range(2)
        ]
        results = evaluate_all_slos(metrics)
        # Should still produce results for all 4 SLOs
        assert len(results) == 4


# =============================================================================
# Integration Test (short baseline run)
# =============================================================================


class TestBaselineIntegration:
    """Integration tests running a short baseline."""

    @pytest.mark.asyncio
    async def test_short_baseline_run(self):
        """A short baseline run completes and produces valid results."""
        result = await run_baseline(
            duration_seconds=2.0,
            concurrency=5,
            num_agents=2,
            num_rounds=1,
        )
        assert result.total_debates > 0
        assert len(result.debate_metrics) > 0
        assert len(result.slo_results) == 4
        assert isinstance(result.all_slos_passed, bool)

    @pytest.mark.asyncio
    async def test_baseline_report_serializable(self):
        """Baseline result can be serialized to valid JSON."""
        result = await run_baseline(
            duration_seconds=1.0,
            concurrency=3,
            num_agents=2,
            num_rounds=1,
        )
        report_dict = result.to_dict()
        json_str = json.dumps(report_dict, indent=2)
        parsed = json.loads(json_str)
        assert parsed["configuration"]["concurrency"] == 3
        assert "slo_validation" in parsed
