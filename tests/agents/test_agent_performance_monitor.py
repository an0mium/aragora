"""
Tests for the Agent Performance Monitor.

Tests cover:
- AgentMetric dataclass
- AgentStats aggregation and calculations
- AgentPerformanceMonitor tracking lifecycle
- Performance insights generation
- Phase breakdown analysis
- Metric persistence and recovery
"""

import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from aragora.agents.performance_monitor import (
    AgentMetric,
    AgentStats,
    AgentPerformanceMonitor,
)


class TestAgentMetric:
    """Tests for the AgentMetric dataclass."""

    def test_create_basic_metric(self):
        """Test creating a basic metric."""
        metric = AgentMetric(
            agent_name="claude",
            operation="generate",
            start_time=1000.0,
        )
        assert metric.agent_name == "claude"
        assert metric.operation == "generate"
        assert metric.start_time == 1000.0
        assert metric.end_time is None
        assert metric.duration_ms is None
        assert metric.success is False
        assert metric.error is None
        assert metric.response_length == 0
        assert metric.phase == ""
        assert metric.round_num == 0

    def test_create_full_metric(self):
        """Test creating a metric with all fields."""
        metric = AgentMetric(
            agent_name="gpt-4",
            operation="critique",
            start_time=1000.0,
            end_time=1005.0,
            duration_ms=5000.0,
            success=True,
            error=None,
            response_length=500,
            phase="debate",
            round_num=2,
        )
        assert metric.agent_name == "gpt-4"
        assert metric.operation == "critique"
        assert metric.duration_ms == 5000.0
        assert metric.success is True
        assert metric.response_length == 500
        assert metric.phase == "debate"
        assert metric.round_num == 2

    def test_metric_with_error(self):
        """Test metric with error information."""
        metric = AgentMetric(
            agent_name="claude",
            operation="generate",
            start_time=1000.0,
            end_time=1010.0,
            duration_ms=10000.0,
            success=False,
            error="Connection timeout",
        )
        assert metric.success is False
        assert metric.error == "Connection timeout"


class TestAgentStats:
    """Tests for the AgentStats class."""

    def test_empty_stats(self):
        """Test initial stats state."""
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

    def test_success_rate_empty(self):
        """Test success rate with no calls."""
        stats = AgentStats()
        assert stats.success_rate == 0.0

    def test_timeout_rate_empty(self):
        """Test timeout rate with no calls."""
        stats = AgentStats()
        assert stats.timeout_rate == 0.0

    def test_update_with_successful_metric(self):
        """Test updating stats with a successful metric."""
        stats = AgentStats()
        metric = AgentMetric(
            agent_name="claude",
            operation="generate",
            start_time=1000.0,
            end_time=1002.0,
            duration_ms=2000.0,
            success=True,
            response_length=100,
        )
        stats.update(metric)

        assert stats.total_calls == 1
        assert stats.successful_calls == 1
        assert stats.failed_calls == 0
        assert stats.total_duration_ms == 2000.0
        assert stats.min_duration_ms == 2000.0
        assert stats.max_duration_ms == 2000.0
        assert stats.avg_duration_ms == 2000.0
        assert stats.total_response_chars == 100
        assert stats.success_rate == 100.0

    def test_update_with_failed_metric(self):
        """Test updating stats with a failed metric."""
        stats = AgentStats()
        metric = AgentMetric(
            agent_name="claude",
            operation="generate",
            start_time=1000.0,
            end_time=1005.0,
            duration_ms=5000.0,
            success=False,
            error="API Error",
            response_length=0,
        )
        stats.update(metric)

        assert stats.total_calls == 1
        assert stats.successful_calls == 0
        assert stats.failed_calls == 1
        assert stats.timeout_calls == 0
        assert stats.success_rate == 0.0

    def test_update_with_timeout_metric(self):
        """Test updating stats with a timeout metric."""
        stats = AgentStats()
        metric = AgentMetric(
            agent_name="claude",
            operation="generate",
            start_time=1000.0,
            end_time=1060.0,
            duration_ms=60000.0,
            success=False,
            error="Request timeout exceeded",
        )
        stats.update(metric)

        assert stats.timeout_calls == 1
        assert stats.timeout_rate == 100.0

    def test_multiple_updates(self):
        """Test multiple metric updates."""
        stats = AgentStats()

        # First call - success
        stats.update(
            AgentMetric(
                agent_name="claude",
                operation="generate",
                start_time=1000.0,
                duration_ms=1000.0,
                success=True,
                response_length=50,
            )
        )

        # Second call - success
        stats.update(
            AgentMetric(
                agent_name="claude",
                operation="generate",
                start_time=1002.0,
                duration_ms=3000.0,
                success=True,
                response_length=150,
            )
        )

        # Third call - failure
        stats.update(
            AgentMetric(
                agent_name="claude",
                operation="generate",
                start_time=1005.0,
                duration_ms=500.0,
                success=False,
                error="Error",
            )
        )

        assert stats.total_calls == 3
        assert stats.successful_calls == 2
        assert stats.failed_calls == 1
        assert stats.total_duration_ms == 4500.0
        assert stats.min_duration_ms == 500.0
        assert stats.max_duration_ms == 3000.0
        assert stats.avg_duration_ms == 1500.0
        assert stats.total_response_chars == 200
        assert stats.success_rate == pytest.approx(66.67, rel=0.01)

    def test_update_with_none_duration(self):
        """Test updating with a metric that has no duration."""
        stats = AgentStats()
        metric = AgentMetric(
            agent_name="claude",
            operation="generate",
            start_time=1000.0,
            duration_ms=None,
            success=True,
            response_length=100,
        )
        stats.update(metric)

        assert stats.total_calls == 1
        assert stats.total_duration_ms == 0.0
        assert stats.min_duration_ms == float("inf")


class TestAgentPerformanceMonitor:
    """Tests for the AgentPerformanceMonitor class."""

    def test_init_without_session_dir(self):
        """Test initialization without session directory."""
        monitor = AgentPerformanceMonitor()
        assert monitor.metrics == []
        assert len(monitor.agent_stats) == 0
        assert monitor.session_dir is None
        assert monitor._active_trackings == {}

    def test_init_with_session_dir(self):
        """Test initialization with session directory."""
        with TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "session"
            monitor = AgentPerformanceMonitor(session_dir=session_dir)
            assert monitor.session_dir == session_dir

    def test_track_agent_call(self):
        """Test starting to track an agent call."""
        monitor = AgentPerformanceMonitor()
        tracking_id = monitor.track_agent_call(
            agent_name="claude",
            operation="generate",
            phase="debate",
            round_num=1,
        )

        assert tracking_id is not None
        assert tracking_id.startswith("claude_generate_")
        assert tracking_id in monitor._active_trackings

        metric = monitor._active_trackings[tracking_id]
        assert metric.agent_name == "claude"
        assert metric.operation == "generate"
        assert metric.phase == "debate"
        assert metric.round_num == 1
        assert metric.start_time > 0

    def test_record_completion_success(self):
        """Test recording successful completion."""
        monitor = AgentPerformanceMonitor()
        tracking_id = monitor.track_agent_call("claude", "generate")

        time.sleep(0.01)  # Small delay to ensure measurable duration

        monitor.record_completion(
            tracking_id,
            success=True,
            response="Hello, world!",
        )

        assert tracking_id not in monitor._active_trackings
        assert len(monitor.metrics) == 1

        metric = monitor.metrics[0]
        assert metric.success is True
        assert metric.error is None
        assert metric.response_length == len("Hello, world!")
        assert metric.duration_ms > 0

        # Check stats updated
        assert "claude" in monitor.agent_stats
        assert monitor.agent_stats["claude"].total_calls == 1
        assert monitor.agent_stats["claude"].successful_calls == 1

    def test_record_completion_failure(self):
        """Test recording failed completion."""
        monitor = AgentPerformanceMonitor()
        tracking_id = monitor.track_agent_call("gpt-4", "critique")

        monitor.record_completion(
            tracking_id,
            success=False,
            error="API rate limit exceeded",
        )

        assert len(monitor.metrics) == 1
        metric = monitor.metrics[0]
        assert metric.success is False
        assert metric.error == "API rate limit exceeded"
        assert metric.response_length == 0

        assert monitor.agent_stats["gpt-4"].failed_calls == 1

    def test_record_completion_unknown_tracking(self):
        """Test recording completion for unknown tracking ID."""
        monitor = AgentPerformanceMonitor()

        # Should not raise, just log warning
        monitor.record_completion(
            "unknown_tracking_id",
            success=True,
            response="test",
        )

        assert len(monitor.metrics) == 0

    def test_record_completion_sanitizes_null_bytes(self):
        """Test that response null bytes are removed."""
        monitor = AgentPerformanceMonitor()
        tracking_id = monitor.track_agent_call("claude", "generate")

        response_with_nulls = "Hello\x00World\x00!"
        monitor.record_completion(
            tracking_id,
            success=True,
            response=response_with_nulls,
        )

        # Null bytes should be removed, so length should be "HelloWorld!" length
        assert monitor.metrics[0].response_length == len("HelloWorld!")

    def test_record_completion_handles_non_string_response(self):
        """Test handling non-string responses."""
        monitor = AgentPerformanceMonitor()
        tracking_id = monitor.track_agent_call("claude", "generate")

        monitor.record_completion(
            tracking_id,
            success=True,
            response=123,  # type: ignore - testing edge case
        )

        assert monitor.metrics[0].response_length == 0

    def test_get_performance_insights_empty(self):
        """Test insights with no metrics."""
        monitor = AgentPerformanceMonitor()
        insights = monitor.get_performance_insights()

        assert "message" in insights
        assert insights["message"] == "No metrics collected yet"

    def test_get_performance_insights_with_data(self):
        """Test insights with collected data."""
        monitor = AgentPerformanceMonitor()

        # Add metrics for multiple agents
        for name in ["claude", "gpt-4", "gemini"]:
            tracking_id = monitor.track_agent_call(name, "generate")
            monitor.record_completion(tracking_id, success=True, response="x" * 100)

        # Add a failure for claude
        tracking_id = monitor.track_agent_call("claude", "critique")
        monitor.record_completion(tracking_id, success=False, error="Error")

        insights = monitor.get_performance_insights()

        assert insights["total_calls"] == 4
        assert "agent_stats" in insights
        assert "claude" in insights["agent_stats"]
        assert "gpt-4" in insights["agent_stats"]
        assert "gemini" in insights["agent_stats"]

        claude_stats = insights["agent_stats"]["claude"]
        assert claude_stats["total_calls"] == 2
        assert claude_stats["success_rate"] == 50.0

    def test_get_performance_insights_slowest_agents(self):
        """Test slowest agents ranking."""
        monitor = AgentPerformanceMonitor()

        # Create agents with different speeds (mock time)
        for name, delay in [("slow", 0.1), ("medium", 0.05), ("fast", 0.01)]:
            tracking_id = monitor.track_agent_call(name, "generate")
            time.sleep(delay)
            monitor.record_completion(tracking_id, success=True, response="ok")

        insights = monitor.get_performance_insights()

        assert len(insights["slowest_agents"]) == 3
        # Slowest should be first
        assert insights["slowest_agents"][0]["agent"] == "slow"

    def test_get_performance_insights_most_failures(self):
        """Test most failures ranking."""
        monitor = AgentPerformanceMonitor()

        # Create agents with different failure rates
        for name, successes, failures in [("reliable", 5, 0), ("flaky", 3, 2), ("broken", 0, 5)]:
            for _ in range(successes):
                tracking_id = monitor.track_agent_call(name, "generate")
                monitor.record_completion(tracking_id, success=True)
            for _ in range(failures):
                tracking_id = monitor.track_agent_call(name, "generate")
                monitor.record_completion(tracking_id, success=False, error="Error")

        insights = monitor.get_performance_insights()

        # Only agents with failures should appear
        failures_list = insights["most_failures"]
        assert len(failures_list) == 2
        assert failures_list[0]["agent"] == "broken"
        assert failures_list[0]["failures"] == 5

    def test_get_performance_insights_recommendations(self):
        """Test recommendations generation."""
        monitor = AgentPerformanceMonitor()

        # Create timeout-prone agent
        for _ in range(5):
            tracking_id = monitor.track_agent_call("timeout_agent", "generate")
            monitor.record_completion(
                tracking_id,
                success=False,
                error="Request timeout",
            )

        insights = monitor.get_performance_insights()

        # Should have timeout recommendation
        recommendations = insights["recommendations"]
        assert len(recommendations) > 0
        assert any("timeout" in r.lower() for r in recommendations)

    def test_get_performance_insights_low_success_recommendation(self):
        """Test low success rate recommendation."""
        monitor = AgentPerformanceMonitor()

        # Create low success rate agent (< 50%)
        for success in [True, False, False, False]:
            tracking_id = monitor.track_agent_call("failing_agent", "generate")
            if success:
                monitor.record_completion(tracking_id, success=True)
            else:
                monitor.record_completion(tracking_id, success=False, error="Error")

        insights = monitor.get_performance_insights()

        recommendations = insights["recommendations"]
        assert any("success rate" in r.lower() for r in recommendations)

    def test_get_phase_breakdown(self):
        """Test phase breakdown analysis."""
        monitor = AgentPerformanceMonitor()

        # Add metrics for different phases
        for phase in ["context", "debate", "debate", "design"]:
            tracking_id = monitor.track_agent_call("claude", "generate", phase=phase)
            monitor.record_completion(tracking_id, success=True, response="ok")

        breakdown = monitor.get_phase_breakdown()

        assert "context" in breakdown
        assert "debate" in breakdown
        assert "design" in breakdown

        assert breakdown["context"]["total_calls"] == 1
        assert breakdown["debate"]["total_calls"] == 2
        assert breakdown["design"]["total_calls"] == 1

    def test_get_phase_breakdown_empty(self):
        """Test phase breakdown with no phases."""
        monitor = AgentPerformanceMonitor()

        # Add metric without phase
        tracking_id = monitor.track_agent_call("claude", "generate", phase="")
        monitor.record_completion(tracking_id, success=True)

        breakdown = monitor.get_phase_breakdown()
        assert len(breakdown) == 0

    def test_save_without_session_dir(self):
        """Test save when no session directory configured."""
        monitor = AgentPerformanceMonitor()
        result = monitor.save()
        assert result is None

    def test_save_with_session_dir(self):
        """Test saving metrics to file."""
        with TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "session"
            monitor = AgentPerformanceMonitor(session_dir=session_dir)

            # Add some metrics
            tracking_id = monitor.track_agent_call("claude", "generate", phase="debate")
            monitor.record_completion(tracking_id, success=True, response="Hello")

            filepath = monitor.save()

            assert filepath is not None
            assert filepath.exists()
            assert filepath.name == "performance_metrics.json"

            # Verify content
            with open(filepath) as f:
                data = json.load(f)

            assert "saved_at" in data
            assert data["metrics_count"] == 1
            assert "insights" in data
            assert "phase_breakdown" in data
            assert "raw_metrics" in data
            assert len(data["raw_metrics"]) == 1

    def test_save_custom_filename(self):
        """Test saving with custom filename."""
        with TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "session"
            monitor = AgentPerformanceMonitor(session_dir=session_dir)

            tracking_id = monitor.track_agent_call("claude", "generate")
            monitor.record_completion(tracking_id, success=True)

            filepath = monitor.save(filename="custom_metrics.json")

            assert filepath is not None
            assert filepath.name == "custom_metrics.json"

    def test_save_limits_raw_metrics(self):
        """Test that save limits raw metrics to last 100."""
        with TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "session"
            monitor = AgentPerformanceMonitor(session_dir=session_dir)

            # Add 150 metrics
            for i in range(150):
                tracking_id = monitor.track_agent_call("claude", "generate")
                monitor.record_completion(tracking_id, success=True, response=str(i))

            filepath = monitor.save()

            with open(filepath) as f:
                data = json.load(f)

            # Should only have last 100
            assert len(data["raw_metrics"]) == 100
            # Last metric should have response_length of "149"
            assert data["raw_metrics"][-1]["response_length"] == 3  # len("149")

    def test_clear(self):
        """Test clearing all metrics."""
        monitor = AgentPerformanceMonitor()

        # Add some data
        tracking_id = monitor.track_agent_call("claude", "generate")
        monitor.record_completion(tracking_id, success=True)

        # Start another tracking (don't complete)
        monitor.track_agent_call("gpt-4", "critique")

        assert len(monitor.metrics) == 1
        assert len(monitor.agent_stats) == 1
        assert len(monitor._active_trackings) == 1

        monitor.clear()

        assert len(monitor.metrics) == 0
        assert len(monitor.agent_stats) == 0
        assert len(monitor._active_trackings) == 0


class TestAgentPerformanceMonitorIntegration:
    """Integration tests for realistic usage scenarios."""

    def test_full_debate_round_tracking(self):
        """Test tracking a full debate round."""
        monitor = AgentPerformanceMonitor()

        agents = ["claude", "gpt-4", "gemini"]
        phases = ["context", "debate", "design"]

        for phase in phases:
            for round_num in range(1, 3):
                for agent in agents:
                    tracking_id = monitor.track_agent_call(
                        agent_name=agent,
                        operation="generate",
                        phase=phase,
                        round_num=round_num,
                    )
                    # Simulate varied response times
                    time.sleep(0.001)
                    monitor.record_completion(
                        tracking_id,
                        success=True,
                        response="x" * (100 + round_num * 10),
                    )

        insights = monitor.get_performance_insights()
        breakdown = monitor.get_phase_breakdown()

        # Should have all agents
        assert len(insights["agent_stats"]) == 3

        # Should have all phases
        assert len(breakdown) == 3

        # Each phase should have 6 calls (3 agents * 2 rounds)
        for phase in phases:
            assert breakdown[phase]["total_calls"] == 6

    def test_mixed_success_failure_scenario(self):
        """Test scenario with mixed successes and failures."""
        monitor = AgentPerformanceMonitor()

        # Reliable agent
        for _ in range(10):
            t = monitor.track_agent_call("reliable", "generate")
            monitor.record_completion(t, success=True, response="ok")

        # Flaky agent (50% success)
        for i in range(10):
            t = monitor.track_agent_call("flaky", "generate")
            if i % 2 == 0:
                monitor.record_completion(t, success=True, response="ok")
            else:
                monitor.record_completion(t, success=False, error="Random failure")

        # Timeout-prone agent
        for i in range(10):
            t = monitor.track_agent_call("slow", "generate")
            if i < 4:  # 40% success
                monitor.record_completion(t, success=True, response="ok")
            else:
                monitor.record_completion(t, success=False, error="Connection timeout")

        insights = monitor.get_performance_insights()

        # Check stats
        assert insights["agent_stats"]["reliable"]["success_rate"] == 100.0
        assert insights["agent_stats"]["flaky"]["success_rate"] == 50.0
        assert insights["agent_stats"]["slow"]["success_rate"] == 40.0

        # Timeout-prone should be identified
        assert len(insights["timeout_prone"]) == 1
        assert insights["timeout_prone"][0]["agent"] == "slow"

        # Should have recommendations for low success rate and timeouts
        assert len(insights["recommendations"]) >= 2

    def test_persistence_and_recovery(self):
        """Test saving and verifying data can be read back."""
        with TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "session"

            # First session - collect metrics
            monitor1 = AgentPerformanceMonitor(session_dir=session_dir)
            for agent in ["claude", "gpt-4"]:
                for _ in range(5):
                    t = monitor1.track_agent_call(agent, "generate", phase="debate")
                    monitor1.record_completion(t, success=True, response="test response")

            filepath = monitor1.save()

            # Load and verify
            with open(filepath) as f:
                saved_data = json.load(f)

            assert saved_data["metrics_count"] == 10
            assert len(saved_data["insights"]["agent_stats"]) == 2
            assert saved_data["phase_breakdown"]["debate"]["total_calls"] == 10

    def test_concurrent_tracking(self):
        """Test handling multiple concurrent trackings."""
        monitor = AgentPerformanceMonitor()

        # Start multiple trackings
        tracking_ids = []
        for agent in ["claude", "gpt-4", "gemini"]:
            t = monitor.track_agent_call(agent, "generate")
            tracking_ids.append((agent, t))

        assert len(monitor._active_trackings) == 3

        # Complete in different order
        for agent, tracking_id in reversed(tracking_ids):
            monitor.record_completion(tracking_id, success=True, response=f"Response from {agent}")

        assert len(monitor._active_trackings) == 0
        assert len(monitor.metrics) == 3

        # All agents should have stats
        assert len(monitor.agent_stats) == 3
