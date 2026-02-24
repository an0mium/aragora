"""Tests for aragora/observability/debate_slos.py -- debate-specific SLO tracking."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from aragora.observability.debate_slos import (
    DebateSLODefinition,
    DebateSLOLevel,
    DebateSLOResult,
    DebateSLOStatus,
    DebateSLOTracker,
    _RollingWindow,
    get_debate_slo_definitions,
    get_debate_slo_status,
    get_debate_slo_tracker,
    reset_debate_slo_tracker,
)


@pytest.fixture(autouse=True)
def _reset_tracker():
    """Reset global tracker between tests."""
    reset_debate_slo_tracker()
    yield
    reset_debate_slo_tracker()


# =============================================================================
# Rolling Window
# =============================================================================


class TestRollingWindow:
    def test_empty_window_returns_zero(self):
        w = _RollingWindow(window_seconds=3600)
        assert w.percentile(95) == 0.0
        assert w.count == 0

    def test_single_value(self):
        w = _RollingWindow(window_seconds=3600)
        w.add(42.0)
        assert w.percentile(50) == 42.0
        assert w.count == 1

    def test_percentile_calculation(self):
        w = _RollingWindow(window_seconds=3600)
        for i in range(100):
            w.add(float(i))
        p50 = w.percentile(50)
        p95 = w.percentile(95)
        p99 = w.percentile(99)
        assert 49 <= p50 <= 51
        assert 94 <= p95 <= 96
        assert 98 <= p99 <= 100

    def test_rate_calculation(self):
        w = _RollingWindow(window_seconds=3600)
        for _ in range(90):
            w.add(1.0)  # success
        for _ in range(10):
            w.add(0.0)  # failure
        rate = w.rate()
        assert abs(rate - 0.9) < 0.01

    def test_rate_empty_returns_one(self):
        w = _RollingWindow(window_seconds=3600)
        assert w.rate() == 1.0

    def test_mean_calculation(self):
        w = _RollingWindow(window_seconds=3600)
        w.add(10.0)
        w.add(20.0)
        w.add(30.0)
        assert w.mean() == 20.0

    def test_max_samples_pruning(self):
        w = _RollingWindow(window_seconds=3600, max_samples=100)
        for i in range(250):
            w.add(float(i))
        # Should prune to max_samples
        assert w.count <= 100


# =============================================================================
# SLO Definitions
# =============================================================================


class TestSLODefinitions:
    def test_get_definitions_returns_five_slos(self):
        defs = get_debate_slo_definitions()
        assert len(defs) == 5
        assert "time_to_first_token" in defs
        assert "debate_completion" in defs
        assert "websocket_reconnection" in defs
        assert "consensus_detection" in defs
        assert "agent_dispatch_concurrency" in defs

    def test_default_targets(self):
        defs = get_debate_slo_definitions()
        assert defs["time_to_first_token"].target == 3.0
        assert defs["debate_completion"].target == 60.0
        assert defs["websocket_reconnection"].target == 0.99
        assert defs["consensus_detection"].target == 0.5
        assert defs["agent_dispatch_concurrency"].target == 0.8

    def test_env_override(self):
        with patch.dict("os.environ", {"SLO_TTFT_P95_S": "5.0"}):
            defs = get_debate_slo_definitions()
            assert defs["time_to_first_token"].target == 5.0

    def test_evaluate_green(self):
        defn = DebateSLODefinition(
            slo_id="test",
            name="Test",
            target=3.0,
            unit="seconds",
            description="Test SLO",
            comparison="lte",
            warning_threshold=4.5,
            critical_threshold=6.0,
        )
        assert defn.evaluate(2.0) == DebateSLOLevel.GREEN

    def test_evaluate_yellow(self):
        defn = DebateSLODefinition(
            slo_id="test",
            name="Test",
            target=3.0,
            unit="seconds",
            description="Test SLO",
            comparison="lte",
            warning_threshold=4.5,
            critical_threshold=6.0,
        )
        assert defn.evaluate(4.0) == DebateSLOLevel.YELLOW

    def test_evaluate_red(self):
        defn = DebateSLODefinition(
            slo_id="test",
            name="Test",
            target=3.0,
            unit="seconds",
            description="Test SLO",
            comparison="lte",
            warning_threshold=4.5,
            critical_threshold=6.0,
        )
        assert defn.evaluate(7.0) == DebateSLOLevel.RED

    def test_evaluate_gte_green(self):
        defn = DebateSLODefinition(
            slo_id="test",
            name="Test",
            target=0.99,
            unit="ratio",
            description="Test SLO",
            comparison="gte",
            warning_threshold=0.985,
            critical_threshold=0.96,
        )
        assert defn.evaluate(0.995) == DebateSLOLevel.GREEN

    def test_evaluate_gte_yellow(self):
        defn = DebateSLODefinition(
            slo_id="test",
            name="Test",
            target=0.99,
            unit="ratio",
            description="Test SLO",
            comparison="gte",
            warning_threshold=0.985,
            critical_threshold=0.96,
        )
        assert defn.evaluate(0.975) == DebateSLOLevel.YELLOW

    def test_evaluate_gte_red(self):
        defn = DebateSLODefinition(
            slo_id="test",
            name="Test",
            target=0.99,
            unit="ratio",
            description="Test SLO",
            comparison="gte",
            warning_threshold=0.985,
            critical_threshold=0.96,
        )
        assert defn.evaluate(0.5) == DebateSLOLevel.RED


# =============================================================================
# Tracker
# =============================================================================


class TestDebateSLOTracker:
    def test_tracker_creation(self):
        tracker = DebateSLOTracker()
        assert tracker is not None

    def test_record_first_token_latency(self):
        tracker = DebateSLOTracker()
        tracker.record_first_token_latency(1.5)
        tracker.record_first_token_latency(2.0)
        status = tracker.get_status("1h")
        slo = status.slos["time_to_first_token"]
        assert slo.sample_count == 2
        assert slo.current > 0

    def test_record_debate_completion(self):
        tracker = DebateSLOTracker()
        tracker.record_debate_completion(30.0, rounds=3, agents=3)
        tracker.record_debate_completion(45.0, rounds=3, agents=3)
        status = tracker.get_status("1h")
        slo = status.slos["debate_completion"]
        assert slo.sample_count == 2

    def test_record_websocket_reconnection(self):
        tracker = DebateSLOTracker()
        for _ in range(99):
            tracker.record_websocket_reconnection(success=True)
        tracker.record_websocket_reconnection(success=False)
        status = tracker.get_status("1h")
        slo = status.slos["websocket_reconnection"]
        assert slo.sample_count == 100
        assert 0.98 <= slo.current <= 1.0

    def test_record_consensus_latency(self):
        tracker = DebateSLOTracker()
        tracker.record_consensus_latency(0.2)
        tracker.record_consensus_latency(0.3)
        status = tracker.get_status("1h")
        slo = status.slos["consensus_detection"]
        assert slo.sample_count == 2

    def test_record_dispatch_concurrency(self):
        tracker = DebateSLOTracker()
        tracker.record_dispatch_concurrency(0.9)
        tracker.record_dispatch_concurrency(0.85)
        status = tracker.get_status("1h")
        slo = status.slos["agent_dispatch_concurrency"]
        assert slo.sample_count == 2

    def test_dispatch_concurrency_clamps(self):
        tracker = DebateSLOTracker()
        tracker.record_dispatch_concurrency(1.5)  # Should clamp to 1.0
        tracker.record_dispatch_concurrency(-0.5)  # Should clamp to 0.0
        status = tracker.get_status("1h")
        slo = status.slos["agent_dispatch_concurrency"]
        # Mean of 1.0 and 0.0 = 0.5
        assert abs(slo.current - 0.5) < 0.01

    def test_overall_healthy_when_all_green(self):
        tracker = DebateSLOTracker()
        # Record good values
        tracker.record_first_token_latency(1.0)
        tracker.record_debate_completion(20.0)
        for _ in range(100):
            tracker.record_websocket_reconnection(success=True)
        tracker.record_consensus_latency(0.1)
        tracker.record_dispatch_concurrency(0.95)

        status = tracker.get_status("1h")
        assert status.overall_healthy is True
        assert status.overall_level == DebateSLOLevel.GREEN

    def test_overall_red_when_any_red(self):
        tracker = DebateSLOTracker()
        # Record one very bad value
        tracker.record_first_token_latency(100.0)  # Way over 3s target
        # Good values for others
        tracker.record_debate_completion(20.0)
        for _ in range(100):
            tracker.record_websocket_reconnection(success=True)
        tracker.record_consensus_latency(0.1)
        tracker.record_dispatch_concurrency(0.95)

        status = tracker.get_status("1h")
        assert status.overall_healthy is False
        assert status.overall_level == DebateSLOLevel.RED

    def test_multi_window_status(self):
        tracker = DebateSLOTracker()
        tracker.record_first_token_latency(1.0)
        result = tracker.get_multi_window_status()
        assert "1h" in result
        assert "24h" in result
        assert "7d" in result
        assert "overall_healthy" in result
        assert "overall_level" in result

    def test_reset(self):
        tracker = DebateSLOTracker()
        tracker.record_first_token_latency(1.0)
        tracker.reset()
        status = tracker.get_status("1h")
        assert status.slos["time_to_first_token"].sample_count == 0


# =============================================================================
# Global Singleton
# =============================================================================


class TestGlobalTracker:
    def test_get_tracker_returns_same_instance(self):
        t1 = get_debate_slo_tracker()
        t2 = get_debate_slo_tracker()
        assert t1 is t2

    def test_reset_creates_new_instance(self):
        t1 = get_debate_slo_tracker()
        reset_debate_slo_tracker()
        t2 = get_debate_slo_tracker()
        assert t1 is not t2

    def test_get_debate_slo_status_convenience(self):
        tracker = get_debate_slo_tracker()
        tracker.record_first_token_latency(1.0)
        status = get_debate_slo_status("1h")
        assert "slos" in status
        assert "timestamp" in status
        assert "overall_healthy" in status


# =============================================================================
# Serialization
# =============================================================================


class TestSerialization:
    def test_result_to_dict(self):
        result = DebateSLOResult(
            slo_id="test",
            name="Test SLO",
            target=3.0,
            current=1.5,
            unit="seconds",
            level=DebateSLOLevel.GREEN,
            compliant=True,
            sample_count=10,
            description="Test description",
        )
        d = result.to_dict()
        assert d["slo_id"] == "test"
        assert d["level"] == "green"
        assert d["compliant"] is True

    def test_status_to_dict(self):
        tracker = DebateSLOTracker()
        tracker.record_first_token_latency(1.0)
        status = tracker.get_status("1h")
        d = status.to_dict()
        assert isinstance(d, dict)
        assert "slos" in d
        assert "time_to_first_token" in d["slos"]
        assert "overall_healthy" in d
        assert "overall_level" in d
        assert "windows" in d
