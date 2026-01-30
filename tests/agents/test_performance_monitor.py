"""
Tests for the Agent Performance Monitor module.

Tests cover:
- AgentMetric dataclass
- AgentStats dataclass and update method
- AgentPerformanceMonitor class
- Tracking, recording, and insights generation
- Phase breakdown analysis
- Save/load functionality
- Error handling and edge cases
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Test module exports."""

    def test_can_import_module(self):
        """Module can be imported."""
        from aragora.agents import performance_monitor

        assert performance_monitor is not None

    def test_agent_metric_in_all(self):
        """AgentMetric is exported in __all__."""
        from aragora.agents.performance_monitor import __all__

        assert "AgentMetric" in __all__

    def test_agent_stats_in_all(self):
        """AgentStats is exported in __all__."""
        from aragora.agents.performance_monitor import __all__

        assert "AgentStats" in __all__

    def test_agent_performance_monitor_in_all(self):
        """AgentPerformanceMonitor is exported in __all__."""
        from aragora.agents.performance_monitor import __all__

        assert "AgentPerformanceMonitor" in __all__


# =============================================================================
# AgentMetric Tests
# =============================================================================


class TestAgentMetricInit:
    """Test AgentMetric initialization."""

    def test_default_values(self):
        """AgentMetric initializes with correct defaults."""
        from aragora.agents.performance_monitor import AgentMetric

        metric = AgentMetric(
            agent_name="test_agent",
            operation="generate",
            start_time=time.time(),
        )

        assert metric.agent_name == "test_agent"
        assert metric.operation == "generate"
        assert metric.end_time is None
        assert metric.duration_ms is None
        assert metric.success is False
        assert metric.error is None
        assert metric.response_length == 0
        assert metric.phase == ""
        assert metric.round_num == 0

    def test_custom_values(self):
        """AgentMetric can be initialized with custom values."""
        from aragora.agents.performance_monitor import AgentMetric

        start = time.time()
        metric = AgentMetric(
            agent_name="claude",
            operation="critique",
            start_time=start,
            end_time=start + 1.0,
            duration_ms=1000.0,
            success=True,
            error=None,
            response_length=500,
            phase="debate",
            round_num=3,
        )

        assert metric.agent_name == "claude"
        assert metric.operation == "critique"
        assert metric.start_time == start
        assert metric.end_time == start + 1.0
        assert metric.duration_ms == 1000.0
        assert metric.success is True
        assert metric.response_length == 500
        assert metric.phase == "debate"
        assert metric.round_num == 3


# =============================================================================
# AgentStats Tests
# =============================================================================


class TestAgentStatsInit:
    """Test AgentStats initialization."""

    def test_default_values(self):
        """AgentStats initializes with correct defaults."""
        from aragora.agents.performance_monitor import AgentStats

        stats = AgentStats()

        assert stats.total_calls == 0
        assert stats.successful_calls == 0
        assert stats.failed_calls == 0
        assert stats.timeout_calls == 0
        assert stats.total_duration_ms == 0.0
        assert stats.min_duration_ms == float("inf")
        assert stats.max_duration_ms == 0.0
        assert stats.avg_duration_ms == 0.0
        assert stats.total_response_chars == 0


class TestAgentStatsUpdate:
    """Test AgentStats.update method."""

    def test_update_successful_call(self):
        """update() correctly tracks successful calls."""
        from aragora.agents.performance_monitor import AgentMetric, AgentStats

        stats = AgentStats()
        metric = AgentMetric(
            agent_name="test",
            operation="generate",
            start_time=time.time(),
            end_time=time.time() + 0.1,
            duration_ms=100.0,
            success=True,
            response_length=250,
        )

        stats.update(metric)

        assert stats.total_calls == 1
        assert stats.successful_calls == 1
        assert stats.failed_calls == 0
        assert stats.total_duration_ms == 100.0
        assert stats.min_duration_ms == 100.0
        assert stats.max_duration_ms == 100.0
        assert stats.avg_duration_ms == 100.0
        assert stats.total_response_chars == 250

    def test_update_failed_call(self):
        """update() correctly tracks failed calls."""
        from aragora.agents.performance_monitor import AgentMetric, AgentStats

        stats = AgentStats()
        metric = AgentMetric(
            agent_name="test",
            operation="generate",
            start_time=time.time(),
            end_time=time.time() + 0.2,
            duration_ms=200.0,
            success=False,
            error="API Error",
        )

        stats.update(metric)

        assert stats.total_calls == 1
        assert stats.successful_calls == 0
        assert stats.failed_calls == 1
        assert stats.timeout_calls == 0

    def test_update_timeout_call(self):
        """update() correctly identifies timeout errors."""
        from aragora.agents.performance_monitor import AgentMetric, AgentStats

        stats = AgentStats()
        metric = AgentMetric(
            agent_name="test",
            operation="generate",
            start_time=time.time(),
            duration_ms=30000.0,
            success=False,
            error="Request timeout after 30 seconds",
        )

        stats.update(metric)

        assert stats.failed_calls == 1
        assert stats.timeout_calls == 1

    def test_update_multiple_calls_tracks_min_max(self):
        """update() correctly tracks min/max durations."""
        from aragora.agents.performance_monitor import AgentMetric, AgentStats

        stats = AgentStats()

        for duration in [100.0, 50.0, 200.0, 75.0]:
            metric = AgentMetric(
                agent_name="test",
                operation="generate",
                start_time=time.time(),
                duration_ms=duration,
                success=True,
            )
            stats.update(metric)

        assert stats.min_duration_ms == 50.0
        assert stats.max_duration_ms == 200.0
        assert stats.avg_duration_ms == 106.25  # (100+50+200+75)/4

    def test_update_handles_none_duration(self):
        """update() handles metrics with None duration."""
        from aragora.agents.performance_monitor import AgentMetric, AgentStats

        stats = AgentStats()
        metric = AgentMetric(
            agent_name="test",
            operation="generate",
            start_time=time.time(),
            duration_ms=None,
            success=True,
        )

        stats.update(metric)

        assert stats.total_calls == 1
        assert stats.total_duration_ms == 0.0
        assert stats.min_duration_ms == float("inf")


class TestAgentStatsProperties:
    """Test AgentStats computed properties."""

    def test_success_rate_no_calls(self):
        """success_rate returns 0 when no calls recorded."""
        from aragora.agents.performance_monitor import AgentStats

        stats = AgentStats()

        assert stats.success_rate == 0.0

    def test_success_rate_calculation(self):
        """success_rate calculates percentage correctly."""
        from aragora.agents.performance_monitor import AgentStats

        stats = AgentStats(
            total_calls=10,
            successful_calls=7,
            failed_calls=3,
        )

        assert stats.success_rate == 70.0

    def test_timeout_rate_no_calls(self):
        """timeout_rate returns 0 when no calls recorded."""
        from aragora.agents.performance_monitor import AgentStats

        stats = AgentStats()

        assert stats.timeout_rate == 0.0

    def test_timeout_rate_calculation(self):
        """timeout_rate calculates percentage correctly."""
        from aragora.agents.performance_monitor import AgentStats

        stats = AgentStats(
            total_calls=20,
            timeout_calls=4,
        )

        assert stats.timeout_rate == 20.0


# =============================================================================
# AgentPerformanceMonitor Initialization Tests
# =============================================================================


class TestAgentPerformanceMonitorInit:
    """Test AgentPerformanceMonitor initialization."""

    def test_init_without_session_dir(self):
        """Monitor initializes without session directory."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()

        assert monitor.metrics == []
        assert monitor.session_dir is None
        assert len(monitor.agent_stats) == 0
        assert len(monitor._active_trackings) == 0

    def test_init_with_session_dir(self, tmp_path):
        """Monitor initializes with session directory."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor(session_dir=tmp_path)

        assert monitor.session_dir == tmp_path


# =============================================================================
# Track Agent Call Tests
# =============================================================================


class TestTrackAgentCall:
    """Test AgentPerformanceMonitor.track_agent_call method."""

    def test_returns_tracking_id(self):
        """track_agent_call returns a tracking ID."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()
        tracking_id = monitor.track_agent_call("claude", "generate")

        assert tracking_id is not None
        assert "claude" in tracking_id
        assert "generate" in tracking_id

    def test_creates_active_tracking(self):
        """track_agent_call creates an active tracking entry."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()
        tracking_id = monitor.track_agent_call("gpt", "critique")

        assert tracking_id in monitor._active_trackings
        metric = monitor._active_trackings[tracking_id]
        assert metric.agent_name == "gpt"
        assert metric.operation == "critique"

    def test_records_phase_and_round(self):
        """track_agent_call stores phase and round information."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()
        tracking_id = monitor.track_agent_call(
            "claude",
            "generate",
            phase="debate",
            round_num=5,
        )

        metric = monitor._active_trackings[tracking_id]
        assert metric.phase == "debate"
        assert metric.round_num == 5

    def test_tracking_id_is_unique(self):
        """Each tracking call generates a unique ID."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()
        id1 = monitor.track_agent_call("claude", "generate")
        time.sleep(0.001)  # Ensure different timestamp
        id2 = monitor.track_agent_call("claude", "generate")

        assert id1 != id2


# =============================================================================
# Record Completion Tests
# =============================================================================


class TestRecordCompletion:
    """Test AgentPerformanceMonitor.record_completion method."""

    def test_records_successful_completion(self):
        """record_completion stores successful call data."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()
        tracking_id = monitor.track_agent_call("claude", "generate")
        time.sleep(0.01)  # Small delay for duration
        monitor.record_completion(tracking_id, success=True, response="Hello world")

        assert len(monitor.metrics) == 1
        metric = monitor.metrics[0]
        assert metric.success is True
        assert metric.duration_ms > 0
        assert metric.response_length == 11
        assert metric.error is None

    def test_records_failed_completion(self):
        """record_completion stores failed call data."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()
        tracking_id = monitor.track_agent_call("gpt", "generate")
        monitor.record_completion(tracking_id, success=False, error="Rate limited")

        metric = monitor.metrics[0]
        assert metric.success is False
        assert metric.error == "Rate limited"

    def test_removes_active_tracking(self):
        """record_completion removes the active tracking entry."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()
        tracking_id = monitor.track_agent_call("claude", "generate")
        assert tracking_id in monitor._active_trackings

        monitor.record_completion(tracking_id, success=True)
        assert tracking_id not in monitor._active_trackings

    def test_updates_agent_stats(self):
        """record_completion updates aggregated agent stats."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()
        tracking_id = monitor.track_agent_call("claude", "generate")
        monitor.record_completion(tracking_id, success=True, response="test")

        stats = monitor.agent_stats["claude"]
        assert stats.total_calls == 1
        assert stats.successful_calls == 1

    def test_handles_unknown_tracking_id(self):
        """record_completion handles unknown tracking IDs gracefully."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()
        # Should not raise
        monitor.record_completion("unknown_id", success=True)

        assert len(monitor.metrics) == 0

    def test_sanitizes_null_bytes_in_response(self):
        """record_completion removes null bytes from response."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()
        tracking_id = monitor.track_agent_call("claude", "generate")
        monitor.record_completion(
            tracking_id,
            success=True,
            response="Hello\0World\0Test",
        )

        metric = monitor.metrics[0]
        assert metric.response_length == 14  # "HelloWorldTest"

    def test_handles_non_string_response(self):
        """record_completion handles non-string response."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()
        tracking_id = monitor.track_agent_call("claude", "generate")
        monitor.record_completion(tracking_id, success=True, response={"key": "value"})

        metric = monitor.metrics[0]
        assert metric.response_length == 0


# =============================================================================
# Performance Insights Tests
# =============================================================================


class TestGetPerformanceInsights:
    """Test AgentPerformanceMonitor.get_performance_insights method."""

    def test_returns_no_metrics_message_when_empty(self):
        """get_performance_insights returns message when no metrics."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()
        insights = monitor.get_performance_insights()

        assert insights == {"message": "No metrics collected yet"}

    def test_returns_basic_stats(self):
        """get_performance_insights returns basic statistics."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()
        for _ in range(5):
            tracking_id = monitor.track_agent_call("claude", "generate")
            monitor.record_completion(tracking_id, success=True, response="test")

        insights = monitor.get_performance_insights()

        assert insights["total_calls"] == 5
        assert "total_duration_ms" in insights
        assert "agent_stats" in insights

    def test_includes_agent_stats(self):
        """get_performance_insights includes per-agent statistics."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()
        for agent in ["claude", "gpt"]:
            tracking_id = monitor.track_agent_call(agent, "generate")
            monitor.record_completion(tracking_id, success=True, response="test")

        insights = monitor.get_performance_insights()

        assert "claude" in insights["agent_stats"]
        assert "gpt" in insights["agent_stats"]
        assert insights["agent_stats"]["claude"]["total_calls"] == 1

    def test_identifies_slowest_agents(self):
        """get_performance_insights ranks agents by response time."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()

        # Create slow and fast agent calls
        tracking_id = monitor.track_agent_call("slow_agent", "generate")
        time.sleep(0.02)
        monitor.record_completion(tracking_id, success=True)

        tracking_id = monitor.track_agent_call("fast_agent", "generate")
        monitor.record_completion(tracking_id, success=True)

        insights = monitor.get_performance_insights()

        assert len(insights["slowest_agents"]) > 0
        assert insights["slowest_agents"][0]["agent"] == "slow_agent"

    def test_identifies_most_failures(self):
        """get_performance_insights identifies agents with most failures."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()

        # Create failing agent
        for _ in range(3):
            tracking_id = monitor.track_agent_call("failing_agent", "generate")
            monitor.record_completion(tracking_id, success=False, error="Error")

        # Create successful agent
        tracking_id = monitor.track_agent_call("good_agent", "generate")
        monitor.record_completion(tracking_id, success=True)

        insights = monitor.get_performance_insights()

        assert len(insights["most_failures"]) > 0
        assert insights["most_failures"][0]["agent"] == "failing_agent"
        assert insights["most_failures"][0]["failures"] == 3

    def test_identifies_timeout_prone_agents(self):
        """get_performance_insights identifies timeout-prone agents."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()

        # Create agent with >20% timeout rate
        for i in range(5):
            tracking_id = monitor.track_agent_call("timeout_agent", "generate")
            if i < 2:
                monitor.record_completion(
                    tracking_id,
                    success=False,
                    error="Connection timeout",
                )
            else:
                monitor.record_completion(tracking_id, success=True)

        insights = monitor.get_performance_insights()

        assert len(insights["timeout_prone"]) > 0
        assert insights["timeout_prone"][0]["agent"] == "timeout_agent"

    def test_generates_recommendations_for_high_timeout(self):
        """get_performance_insights generates recommendations for timeout issues."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()

        # Create agent with >30% timeout rate
        for i in range(10):
            tracking_id = monitor.track_agent_call("bad_agent", "generate")
            if i < 4:
                monitor.record_completion(
                    tracking_id,
                    success=False,
                    error="Request timeout",
                )
            else:
                monitor.record_completion(tracking_id, success=True)

        insights = monitor.get_performance_insights()

        assert any("timeout" in r.lower() for r in insights["recommendations"])

    def test_generates_recommendations_for_slow_agents(self):
        """get_performance_insights recommends optimization for slow agents."""
        from aragora.agents.performance_monitor import AgentMetric, AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()

        # Manually create a metric with very long duration
        metric = AgentMetric(
            agent_name="slow_agent",
            operation="generate",
            start_time=time.time() - 70,
            end_time=time.time(),
            duration_ms=70000,  # >60 seconds
            success=True,
        )
        monitor.metrics.append(metric)
        monitor.agent_stats["slow_agent"].update(metric)

        insights = monitor.get_performance_insights()

        assert any("slow_agent" in r and "averages" in r for r in insights["recommendations"])

    def test_generates_recommendations_for_low_success_rate(self):
        """get_performance_insights recommends checking agents with low success."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()

        # Create agent with <50% success rate (3 calls minimum)
        for i in range(4):
            tracking_id = monitor.track_agent_call("unreliable_agent", "generate")
            monitor.record_completion(
                tracking_id,
                success=(i == 0),  # Only 1/4 success
                error=None if i == 0 else "Error",
            )

        insights = monitor.get_performance_insights()

        assert any("unreliable_agent" in r for r in insights["recommendations"])


# =============================================================================
# Phase Breakdown Tests
# =============================================================================


class TestGetPhaseBreakdown:
    """Test AgentPerformanceMonitor.get_phase_breakdown method."""

    def test_returns_empty_for_no_metrics(self):
        """get_phase_breakdown returns empty dict when no metrics."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()
        breakdown = monitor.get_phase_breakdown()

        assert breakdown == {}

    def test_groups_metrics_by_phase(self):
        """get_phase_breakdown groups metrics by debate phase."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()

        for phase in ["context", "debate", "design"]:
            tracking_id = monitor.track_agent_call("claude", "generate", phase=phase)
            monitor.record_completion(tracking_id, success=True)

        breakdown = monitor.get_phase_breakdown()

        assert "context" in breakdown
        assert "debate" in breakdown
        assert "design" in breakdown

    def test_calculates_phase_stats(self):
        """get_phase_breakdown calculates correct statistics per phase."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()

        # Add 3 calls to debate phase, 2 successful
        for i in range(3):
            tracking_id = monitor.track_agent_call("claude", "generate", phase="debate")
            monitor.record_completion(tracking_id, success=(i < 2))

        breakdown = monitor.get_phase_breakdown()

        assert breakdown["debate"]["total_calls"] == 3
        assert abs(breakdown["debate"]["success_rate"] - 66.7) < 0.1

    def test_skips_metrics_without_phase(self):
        """get_phase_breakdown skips metrics without phase."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()

        # Metric without phase
        tracking_id = monitor.track_agent_call("claude", "generate", phase="")
        monitor.record_completion(tracking_id, success=True)

        # Metric with phase
        tracking_id = monitor.track_agent_call("claude", "generate", phase="debate")
        monitor.record_completion(tracking_id, success=True)

        breakdown = monitor.get_phase_breakdown()

        assert "" not in breakdown
        assert "debate" in breakdown


# =============================================================================
# Save Tests
# =============================================================================


class TestSave:
    """Test AgentPerformanceMonitor.save method."""

    def test_returns_none_without_session_dir(self):
        """save() returns None when no session_dir configured."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()
        result = monitor.save()

        assert result is None

    def test_creates_directory_if_needed(self, tmp_path):
        """save() creates session directory if it doesn't exist."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        session_dir = tmp_path / "metrics_subdir"
        monitor = AgentPerformanceMonitor(session_dir=session_dir)

        tracking_id = monitor.track_agent_call("claude", "generate")
        monitor.record_completion(tracking_id, success=True)

        result = monitor.save()

        assert result is not None
        assert session_dir.exists()

    def test_saves_json_file(self, tmp_path):
        """save() creates a valid JSON file."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor(session_dir=tmp_path)
        tracking_id = monitor.track_agent_call("claude", "generate", phase="debate")
        monitor.record_completion(tracking_id, success=True, response="test response")

        filepath = monitor.save()

        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)

        assert "saved_at" in data
        assert "metrics_count" in data
        assert "insights" in data
        assert "phase_breakdown" in data
        assert "raw_metrics" in data

    def test_saves_with_custom_filename(self, tmp_path):
        """save() uses custom filename when provided."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor(session_dir=tmp_path)
        tracking_id = monitor.track_agent_call("claude", "generate")
        monitor.record_completion(tracking_id, success=True)

        filepath = monitor.save(filename="custom_metrics.json")

        assert filepath.name == "custom_metrics.json"

    def test_limits_raw_metrics_to_100(self, tmp_path):
        """save() only stores last 100 raw metrics."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor(session_dir=tmp_path)

        # Add 150 metrics
        for _ in range(150):
            tracking_id = monitor.track_agent_call("claude", "generate")
            monitor.record_completion(tracking_id, success=True)

        filepath = monitor.save()

        with open(filepath) as f:
            data = json.load(f)

        assert data["metrics_count"] == 150
        assert len(data["raw_metrics"]) == 100

    def test_handles_save_error(self, tmp_path):
        """save() returns None on write error."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor(session_dir=tmp_path)
        tracking_id = monitor.track_agent_call("claude", "generate")
        monitor.record_completion(tracking_id, success=True)

        with patch("builtins.open", side_effect=OSError("Permission denied")):
            result = monitor.save()

        assert result is None


# =============================================================================
# Clear Tests
# =============================================================================


class TestClear:
    """Test AgentPerformanceMonitor.clear method."""

    def test_clears_all_data(self):
        """clear() removes all collected data."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()

        # Add some data
        tracking_id = monitor.track_agent_call("claude", "generate")
        monitor.record_completion(tracking_id, success=True)

        # Start another tracking (not completed)
        monitor.track_agent_call("gpt", "critique")

        monitor.clear()

        assert len(monitor.metrics) == 0
        assert len(monitor.agent_stats) == 0
        assert len(monitor._active_trackings) == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestPerformanceMonitorIntegration:
    """Integration tests for full performance monitoring workflow."""

    def test_full_tracking_workflow(self):
        """Test complete tracking workflow."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()

        # Track multiple agents across phases
        agents = ["claude", "gpt", "gemini"]
        phases = ["context", "debate", "design"]

        for agent in agents:
            for phase in phases:
                tracking_id = monitor.track_agent_call(
                    agent,
                    "generate",
                    phase=phase,
                    round_num=1,
                )
                # Simulate some work
                time.sleep(0.001)
                monitor.record_completion(
                    tracking_id,
                    success=(agent != "gpt"),  # GPT fails
                    response="Response text" if agent != "gpt" else None,
                    error="API Error" if agent == "gpt" else None,
                )

        # Verify metrics
        assert len(monitor.metrics) == 9

        # Check insights
        insights = monitor.get_performance_insights()
        assert insights["total_calls"] == 9
        assert "claude" in insights["agent_stats"]
        assert insights["agent_stats"]["claude"]["success_rate"] == 100.0
        assert insights["agent_stats"]["gpt"]["success_rate"] == 0.0

        # Check phase breakdown
        breakdown = monitor.get_phase_breakdown()
        assert len(breakdown) == 3
        for phase in phases:
            assert breakdown[phase]["total_calls"] == 3

    def test_save_and_reload_workflow(self, tmp_path):
        """Test saving metrics and verifying persisted data."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor(session_dir=tmp_path)

        # Track some calls
        for i in range(5):
            tracking_id = monitor.track_agent_call(
                "claude",
                "generate",
                phase="debate",
                round_num=i,
            )
            monitor.record_completion(
                tracking_id,
                success=(i % 2 == 0),
                response="Response" if i % 2 == 0 else None,
                error="Error" if i % 2 != 0 else None,
            )

        # Save
        filepath = monitor.save()

        # Reload and verify
        with open(filepath) as f:
            data = json.load(f)

        assert data["metrics_count"] == 5
        assert len(data["raw_metrics"]) == 5
        assert data["insights"]["total_calls"] == 5
        assert "debate" in data["phase_breakdown"]


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_concurrent_tracking_multiple_agents(self):
        """Multiple agents can be tracked concurrently."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()

        # Start multiple trackings
        id1 = monitor.track_agent_call("claude", "generate")
        id2 = monitor.track_agent_call("gpt", "critique")
        id3 = monitor.track_agent_call("gemini", "vote")

        assert len(monitor._active_trackings) == 3

        # Complete in different order
        monitor.record_completion(id2, success=True)
        monitor.record_completion(id3, success=True)
        monitor.record_completion(id1, success=True)

        assert len(monitor._active_trackings) == 0
        assert len(monitor.metrics) == 3

    def test_empty_response_handling(self):
        """Empty response is handled correctly."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()
        tracking_id = monitor.track_agent_call("claude", "generate")
        monitor.record_completion(tracking_id, success=True, response="")

        metric = monitor.metrics[0]
        assert metric.response_length == 0

    def test_none_response_handling(self):
        """None response is handled correctly."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()
        tracking_id = monitor.track_agent_call("claude", "generate")
        monitor.record_completion(tracking_id, success=True, response=None)

        metric = monitor.metrics[0]
        assert metric.response_length == 0

    def test_special_characters_in_agent_name(self):
        """Agent names with special characters are handled."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()
        agent_name = "claude-3.5-sonnet@anthropic"
        tracking_id = monitor.track_agent_call(agent_name, "generate")
        monitor.record_completion(tracking_id, success=True)

        assert agent_name in monitor.agent_stats

    def test_very_long_response(self):
        """Very long responses are measured correctly."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()
        tracking_id = monitor.track_agent_call("claude", "generate")
        long_response = "x" * 1000000  # 1MB
        monitor.record_completion(tracking_id, success=True, response=long_response)

        metric = monitor.metrics[0]
        assert metric.response_length == 1000000

    def test_insights_with_min_duration_inf(self):
        """Insights handle agents with no duration data."""
        from aragora.agents.performance_monitor import AgentMetric, AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()

        # Add metric with no duration
        metric = AgentMetric(
            agent_name="test",
            operation="generate",
            start_time=time.time(),
            duration_ms=None,
            success=True,
        )
        monitor.metrics.append(metric)
        monitor.agent_stats["test"].update(metric)

        insights = monitor.get_performance_insights()

        # min_duration_ms should be 0 (not inf) in output
        assert insights["agent_stats"]["test"]["min_duration_ms"] == 0

    def test_record_completion_twice_same_id(self):
        """Recording completion twice with same ID is handled."""
        from aragora.agents.performance_monitor import AgentPerformanceMonitor

        monitor = AgentPerformanceMonitor()
        tracking_id = monitor.track_agent_call("claude", "generate")
        monitor.record_completion(tracking_id, success=True)

        # Second call should be ignored (ID already removed)
        monitor.record_completion(tracking_id, success=False)

        assert len(monitor.metrics) == 1
        assert monitor.metrics[0].success is True
