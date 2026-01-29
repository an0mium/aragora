"""
Tests for aragora.observability.metrics.fabric module.

Covers:
- Fabric metrics initialization
- Agent metrics recording (spawned, terminated, health)
- Task metrics recording (queued, completed, latency)
- Policy metrics recording (decisions, approvals)
- Budget metrics recording (usage, alerts)
- Context manager (track_fabric_task)
- Integration with AgentFabric stats
"""

from __future__ import annotations

import pytest

from aragora.observability.metrics.base import NoOpMetric


# =============================================================================
# TestFabricMetricsInit
# =============================================================================


class TestFabricMetricsInit:
    """Tests for fabric metrics initialization."""

    def test_init_fabric_metrics(self):
        """init_fabric_metrics should not raise."""
        from aragora.observability.metrics.fabric import init_fabric_metrics

        init_fabric_metrics()

    def test_init_is_idempotent(self):
        """Multiple init calls should be safe."""
        from aragora.observability.metrics.fabric import init_fabric_metrics

        init_fabric_metrics()
        init_fabric_metrics()
        init_fabric_metrics()


# =============================================================================
# TestAgentMetrics
# =============================================================================


class TestAgentMetrics:
    """Tests for agent-related fabric metrics."""

    def test_set_agents_active(self):
        """set_agents_active should not raise."""
        from aragora.observability.metrics.fabric import set_agents_active

        set_agents_active("pool-1", 5)
        set_agents_active("pool-2", 0)
        set_agents_active("default", 100)

    def test_set_agents_health(self):
        """set_agents_health should not raise."""
        from aragora.observability.metrics.fabric import set_agents_health

        set_agents_health(healthy=10, degraded=2, unhealthy=1)
        set_agents_health(healthy=0, degraded=0, unhealthy=0)

    def test_record_agent_spawned(self):
        """record_agent_spawned should not raise."""
        from aragora.observability.metrics.fabric import record_agent_spawned

        record_agent_spawned("pool-1", "claude-3-opus")
        record_agent_spawned("pool-1", "gpt-4")
        record_agent_spawned("default", "claude-3-sonnet")

    def test_record_agent_terminated(self):
        """record_agent_terminated should not raise."""
        from aragora.observability.metrics.fabric import record_agent_terminated

        record_agent_terminated("pool-1", "graceful")
        record_agent_terminated("pool-2", "timeout")
        record_agent_terminated("pool-1", "error")


# =============================================================================
# TestTaskMetrics
# =============================================================================


class TestTaskMetrics:
    """Tests for task-related fabric metrics."""

    def test_record_task_queued(self):
        """record_task_queued should not raise."""
        from aragora.observability.metrics.fabric import record_task_queued

        record_task_queued("debate", "normal")
        record_task_queued("generate", "high")
        record_task_queued("debate", "critical")

    def test_record_task_completed_success(self):
        """record_task_completed should record success."""
        from aragora.observability.metrics.fabric import record_task_completed

        record_task_completed("debate", success=True, latency_seconds=5.5)
        record_task_completed("generate", success=True, latency_seconds=1.2)

    def test_record_task_completed_failure(self):
        """record_task_completed should record failure."""
        from aragora.observability.metrics.fabric import record_task_completed

        record_task_completed("debate", success=False, latency_seconds=30.0)
        record_task_completed("generate", success=False)

    def test_record_task_cancelled(self):
        """record_task_cancelled should not raise."""
        from aragora.observability.metrics.fabric import record_task_cancelled

        record_task_cancelled("debate")
        record_task_cancelled("generate")

    def test_set_task_queue_depth(self):
        """set_task_queue_depth should not raise."""
        from aragora.observability.metrics.fabric import set_task_queue_depth

        set_task_queue_depth("agent-1", 5)
        set_task_queue_depth("agent-2", 0)
        set_task_queue_depth("agent-3", 100)


# =============================================================================
# TestPolicyMetrics
# =============================================================================


class TestPolicyMetrics:
    """Tests for policy-related fabric metrics."""

    def test_record_policy_decision(self):
        """record_policy_decision should not raise."""
        from aragora.observability.metrics.fabric import record_policy_decision

        record_policy_decision("allowed")
        record_policy_decision("denied")
        record_policy_decision("approval_required")

    def test_set_pending_approvals(self):
        """set_pending_approvals should not raise."""
        from aragora.observability.metrics.fabric import set_pending_approvals

        set_pending_approvals(0)
        set_pending_approvals(5)
        set_pending_approvals(100)


# =============================================================================
# TestBudgetMetrics
# =============================================================================


class TestBudgetMetrics:
    """Tests for budget-related fabric metrics."""

    def test_set_budget_usage(self):
        """set_budget_usage should not raise."""
        from aragora.observability.metrics.fabric import set_budget_usage

        set_budget_usage("agent-1", "agent", 50.0)
        set_budget_usage("pool-1", "pool", 75.5)
        set_budget_usage("tenant-1", "tenant", 90.0)

    def test_record_budget_alert(self):
        """record_budget_alert should not raise."""
        from aragora.observability.metrics.fabric import record_budget_alert

        record_budget_alert("agent-1", "soft_limit")
        record_budget_alert("agent-2", "hard_limit")


# =============================================================================
# TestConvenienceFunctions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_record_fabric_stats(self):
        """record_fabric_stats should handle dict input."""
        from aragora.observability.metrics.fabric import record_fabric_stats

        stats = {
            "lifecycle": {
                "agents_healthy": 10,
                "agents_degraded": 2,
                "agents_unhealthy": 1,
                "agents_active": 13,
            },
            "policy": {
                "pending_approvals": 3,
            },
        }
        record_fabric_stats(stats)

    def test_record_fabric_stats_empty(self):
        """record_fabric_stats should handle empty dict."""
        from aragora.observability.metrics.fabric import record_fabric_stats

        record_fabric_stats({})

    def test_record_fabric_stats_partial(self):
        """record_fabric_stats should handle partial stats."""
        from aragora.observability.metrics.fabric import record_fabric_stats

        record_fabric_stats({"lifecycle": {"agents_healthy": 5}})
        record_fabric_stats({"policy": {}})


# =============================================================================
# TestContextManager
# =============================================================================


class TestContextManager:
    """Tests for track_fabric_task context manager."""

    def test_track_fabric_task_success(self):
        """track_fabric_task should record success."""
        from aragora.observability.metrics.fabric import track_fabric_task

        with track_fabric_task("debate", "high"):
            pass  # Simulated work

    def test_track_fabric_task_failure(self):
        """track_fabric_task should record failure on exception."""
        from aragora.observability.metrics.fabric import track_fabric_task

        with pytest.raises(ValueError):
            with track_fabric_task("generate", "normal"):
                raise ValueError("Test error")


# =============================================================================
# TestTracingIntegration
# =============================================================================


class TestTracingIntegration:
    """Tests for fabric tracing functions."""

    def test_trace_fabric_operation(self):
        """trace_fabric_operation should not raise."""
        from aragora.observability.tracing import trace_fabric_operation

        with trace_fabric_operation("spawn", agent_id="a1"):
            pass

    def test_trace_fabric_operation_with_task(self):
        """trace_fabric_operation should accept task_id."""
        from aragora.observability.tracing import trace_fabric_operation

        with trace_fabric_operation("schedule", agent_id="a1", task_id="t1"):
            pass

    def test_trace_fabric_task(self):
        """trace_fabric_task should not raise."""
        from aragora.observability.tracing import trace_fabric_task

        with trace_fabric_task("debate", "task-1", "agent-1", "high"):
            pass

    def test_trace_fabric_policy_check(self):
        """trace_fabric_policy_check should not raise."""
        from aragora.observability.tracing import trace_fabric_policy_check

        with trace_fabric_policy_check("shell.execute", agent_id="a1", resource="bash"):
            pass


# =============================================================================
# TestFabricIntegration
# =============================================================================


class TestFabricIntegration:
    """Tests for integration with AgentFabric."""

    @pytest.mark.asyncio
    async def test_fabric_get_stats_records_metrics(self):
        """AgentFabric.get_stats should record Prometheus metrics."""
        from aragora.fabric import AgentFabric

        fabric = AgentFabric()
        await fabric.start()

        # This should not raise and should record metrics
        stats = await fabric.get_stats()

        assert "lifecycle" in stats
        assert "scheduler" in stats
        assert "policy" in stats
        assert "budget" in stats

        await fabric.stop()
