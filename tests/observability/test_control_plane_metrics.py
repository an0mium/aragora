"""Tests for control plane metrics module."""

import pytest

from aragora.observability.metrics.control_plane import (
    _NoOpMetric,
    record_agent_heartbeat,
    record_agent_health_check,
    record_agent_registered,
    record_deliberation_completed,
    record_deliberation_sla,
    record_deliberation_started,
    record_policy_decision,
    record_policy_violation,
    record_task_completed,
    record_task_failed,
    record_task_submitted,
    record_task_wait_time,
    set_active_agents,
    set_task_queue_depth,
)


class TestNoOpMetric:
    """Test NoOpMetric fallback class."""

    def test_labels_returns_self(self):
        """Test labels() returns self for chaining."""
        metric = _NoOpMetric()
        result = metric.labels("label1", label2="value")
        assert result is metric

    def test_inc_does_nothing(self):
        """Test inc() completes without error."""
        metric = _NoOpMetric()
        metric.inc()
        metric.inc(5)

    def test_dec_does_nothing(self):
        """Test dec() completes without error."""
        metric = _NoOpMetric()
        metric.dec()
        metric.dec(3)

    def test_set_does_nothing(self):
        """Test set() completes without error."""
        metric = _NoOpMetric()
        metric.set(100)

    def test_observe_does_nothing(self):
        """Test observe() completes without error."""
        metric = _NoOpMetric()
        metric.observe(45.5)


class TestAgentMetrics:
    """Test agent-related metrics recording."""

    def test_record_agent_registered_success(self):
        """Test recording successful agent registration."""
        record_agent_registered("agent-001", success=True)
        # Should not raise

    def test_record_agent_registered_failure(self):
        """Test recording failed agent registration."""
        record_agent_registered("agent-002", success=False)
        # Should not raise

    def test_set_active_agents(self):
        """Test setting active agent count."""
        set_active_agents("debate", 5)
        set_active_agents("summarize", 3)
        # Should not raise

    def test_record_agent_heartbeat(self):
        """Test recording agent heartbeat."""
        record_agent_heartbeat("agent-001", "healthy")
        record_agent_heartbeat("agent-002", "degraded")
        record_agent_heartbeat("agent-003", "unhealthy")
        # Should not raise

    def test_record_agent_health_check(self):
        """Test recording agent health check."""
        record_agent_health_check("healthy")
        record_agent_health_check("unhealthy")
        record_agent_health_check("timeout")
        # Should not raise


class TestTaskMetrics:
    """Test task-related metrics recording."""

    def test_record_task_submitted(self):
        """Test recording task submission."""
        record_task_submitted("deliberation")
        record_task_submitted("summarization", priority="high")
        # Should not raise

    def test_record_task_completed(self):
        """Test recording task completion."""
        record_task_completed("deliberation", duration_seconds=45.5)
        record_task_completed("summarization", duration_seconds=12.3)
        # Should not raise

    def test_record_task_failed(self):
        """Test recording task failure."""
        record_task_failed("deliberation", "timeout")
        record_task_failed("summarization", "agent_unavailable")
        # Should not raise

    def test_set_task_queue_depth(self):
        """Test setting task queue depth."""
        set_task_queue_depth("normal", 10)
        set_task_queue_depth("high", 3)
        set_task_queue_depth("low", 25)
        # Should not raise

    def test_record_task_wait_time(self):
        """Test recording task wait time."""
        record_task_wait_time(0.5)
        record_task_wait_time(30.0)
        record_task_wait_time(120.0)
        # Should not raise


class TestDeliberationMetrics:
    """Test deliberation-related metrics recording."""

    def test_record_deliberation_started(self):
        """Test recording deliberation start."""
        record_deliberation_started()
        record_deliberation_started(mode="async")
        record_deliberation_started(mode="scheduled")
        # Should not raise

    def test_record_deliberation_completed_success(self):
        """Test recording successful deliberation."""
        record_deliberation_completed(
            success=True,
            consensus_reached=True,
            duration_seconds=45.5,
            consensus_confidence=0.85,
            agent_count=3,
        )
        # Should not raise

    def test_record_deliberation_completed_no_consensus(self):
        """Test recording deliberation without consensus."""
        record_deliberation_completed(
            success=True,
            consensus_reached=False,
            duration_seconds=120.0,
        )
        # Should not raise

    def test_record_deliberation_completed_failure(self):
        """Test recording failed deliberation."""
        record_deliberation_completed(
            success=False,
            consensus_reached=False,
            duration_seconds=30.0,
        )
        # Should not raise

    def test_record_deliberation_completed_minimal(self):
        """Test recording with minimal parameters."""
        record_deliberation_completed(
            success=True,
            consensus_reached=True,
            duration_seconds=10.0,
        )
        # Should not raise

    def test_record_deliberation_sla(self):
        """Test recording SLA compliance levels."""
        record_deliberation_sla("compliant")
        record_deliberation_sla("warning")
        record_deliberation_sla("critical")
        record_deliberation_sla("violated")
        # Should not raise


class TestPolicyMetrics:
    """Test policy-related metrics recording."""

    def test_record_policy_decision_allow(self):
        """Test recording allow policy decision."""
        record_policy_decision("task_dispatch", "allow")
        # Should not raise

    def test_record_policy_decision_deny(self):
        """Test recording deny policy decision."""
        record_policy_decision("agent_access", "deny")
        # Should not raise

    def test_record_policy_decision_with_latency(self):
        """Test recording policy decision with latency."""
        record_policy_decision("task_dispatch", "allow", latency_seconds=0.005)
        # Should not raise

    def test_record_policy_violation(self):
        """Test recording policy violation."""
        record_policy_violation("rate_limit", "warning")
        record_policy_violation("agent_allowlist", "critical")
        # Should not raise


class TestMetricsInitialization:
    """Test metrics initialization behavior."""

    def test_metrics_initialize_on_first_call(self):
        """Test metrics lazy initialization."""
        # First call should trigger initialization
        record_agent_registered("test-agent", success=True)
        # Subsequent calls should reuse initialized metrics
        record_agent_registered("test-agent-2", success=True)
        # Should not raise

    def test_multiple_metric_types_can_coexist(self):
        """Test different metric types work together."""
        record_agent_registered("agent-1", success=True)
        record_task_submitted("deliberation")
        record_deliberation_started()
        record_policy_decision("access", "allow")
        # Should not raise


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_agent_id(self):
        """Test with empty agent ID."""
        record_agent_registered("", success=True)
        # Should not raise

    def test_very_long_task_type(self):
        """Test with very long task type string."""
        long_type = "a" * 1000
        record_task_submitted(long_type)
        # Should not raise

    def test_zero_duration(self):
        """Test with zero duration."""
        record_task_completed("quick_task", duration_seconds=0.0)
        # Should not raise

    def test_negative_queue_depth(self):
        """Test with negative queue depth (edge case)."""
        set_task_queue_depth("normal", -1)
        # Should not raise (gauge can be negative technically)

    def test_high_precision_latency(self):
        """Test with high precision latency value."""
        record_policy_decision("test", "allow", latency_seconds=0.000123456789)
        # Should not raise

    def test_very_high_confidence(self):
        """Test with confidence at boundary."""
        record_deliberation_completed(
            success=True,
            consensus_reached=True,
            duration_seconds=10.0,
            consensus_confidence=1.0,
        )
        # Should not raise

    def test_unicode_in_labels(self):
        """Test with unicode characters in labels."""
        record_agent_heartbeat("agent-日本語", "healthy")
        record_task_submitted("タスク")
        # Should not raise
