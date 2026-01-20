"""
Tests for persistence store metrics.

Tests that metrics functions are callable and work correctly
with and without prometheus_client installed.
"""

import pytest

from aragora.observability.metrics import (
    _init_metrics,
    _init_noop_metrics,
    # Task Queue metrics
    record_task_queue_operation,
    set_task_queue_size,
    record_task_queue_recovery,
    record_task_queue_cleanup,
    track_task_queue_operation,
    # Governance metrics
    record_governance_decision,
    record_governance_verification,
    record_governance_approval,
    record_governance_store_latency,
    set_governance_artifacts_active,
    track_governance_store_operation,
    # User mapping metrics
    record_user_mapping_operation,
    record_user_mapping_cache_hit,
    record_user_mapping_cache_miss,
    set_user_mappings_active,
)


class TestTaskQueueMetrics:
    """Tests for persistent task queue metrics."""

    def test_record_task_queue_operation(self):
        """Should record task queue operation without error."""
        _init_noop_metrics()
        # Should not raise
        record_task_queue_operation("enqueue", True, 0.01)
        record_task_queue_operation("dequeue", True, 0.005)
        record_task_queue_operation("complete", True, 0.002)
        record_task_queue_operation("fail", False, 0.001)

    def test_set_task_queue_size(self):
        """Should set task queue size gauges."""
        _init_noop_metrics()
        set_task_queue_size(pending=10, ready=5, running=3)

    def test_record_task_queue_recovery(self):
        """Should record recovery metrics."""
        _init_noop_metrics()
        record_task_queue_recovery("pending", 5)
        record_task_queue_recovery("ready", 3)
        record_task_queue_recovery("running", 2)

    def test_record_task_queue_cleanup(self):
        """Should record cleanup metrics."""
        _init_noop_metrics()
        record_task_queue_cleanup(10)

    def test_track_task_queue_operation_success(self):
        """Should track operation latency on success."""
        _init_noop_metrics()
        with track_task_queue_operation("enqueue"):
            pass  # Simulate work

    def test_track_task_queue_operation_failure(self):
        """Should track operation failure."""
        _init_noop_metrics()
        with pytest.raises(ValueError):
            with track_task_queue_operation("enqueue"):
                raise ValueError("Test error")


class TestGovernanceMetrics:
    """Tests for governance store metrics."""

    def test_record_governance_decision(self):
        """Should record governance decisions."""
        _init_noop_metrics()
        record_governance_decision("auto", "approved")
        record_governance_decision("manual", "rejected")

    def test_record_governance_verification(self):
        """Should record verifications."""
        _init_noop_metrics()
        record_governance_verification("formal", "valid")
        record_governance_verification("runtime", "invalid")

    def test_record_governance_approval(self):
        """Should record approvals."""
        _init_noop_metrics()
        record_governance_approval("nomic", "granted")
        record_governance_approval("deploy", "revoked")

    def test_record_governance_store_latency(self):
        """Should record store operation latency."""
        _init_noop_metrics()
        record_governance_store_latency("save", 0.005)
        record_governance_store_latency("get", 0.001)
        record_governance_store_latency("list", 0.010)
        record_governance_store_latency("delete", 0.003)

    def test_set_governance_artifacts_active(self):
        """Should set active artifact counts."""
        _init_noop_metrics()
        set_governance_artifacts_active(
            decisions=100,
            verifications=50,
            approvals=25,
        )

    def test_track_governance_store_operation(self):
        """Should track store operation latency."""
        _init_noop_metrics()
        with track_governance_store_operation("save"):
            pass  # Simulate work


class TestUserMappingMetrics:
    """Tests for user ID mapping metrics."""

    def test_record_user_mapping_operation(self):
        """Should record mapping operations."""
        _init_noop_metrics()
        record_user_mapping_operation("save", "slack", True)
        record_user_mapping_operation("get", "discord", True)
        record_user_mapping_operation("get", "teams", False)
        record_user_mapping_operation("delete", "slack", True)

    def test_record_user_mapping_cache_hit(self):
        """Should record cache hits."""
        _init_noop_metrics()
        record_user_mapping_cache_hit("slack")
        record_user_mapping_cache_hit("discord")

    def test_record_user_mapping_cache_miss(self):
        """Should record cache misses."""
        _init_noop_metrics()
        record_user_mapping_cache_miss("slack")
        record_user_mapping_cache_miss("teams")

    def test_set_user_mappings_active(self):
        """Should set active mapping counts."""
        _init_noop_metrics()
        set_user_mappings_active("slack", 100)
        set_user_mappings_active("discord", 50)
        set_user_mappings_active("teams", 25)


class TestMetricsInitialization:
    """Tests for metrics initialization."""

    def test_noop_metrics_all_functions_work(self):
        """All metric functions should work with NoOp metrics."""
        _init_noop_metrics()

        # All should execute without error
        record_task_queue_operation("test", True, 0.001)
        set_task_queue_size(1, 2, 3)
        record_task_queue_recovery("pending", 1)
        record_task_queue_cleanup(1)

        record_governance_decision("auto", "approved")
        record_governance_verification("formal", "valid")
        record_governance_approval("nomic", "granted")
        record_governance_store_latency("save", 0.001)
        set_governance_artifacts_active(1, 2, 3)

        record_user_mapping_operation("save", "slack", True)
        record_user_mapping_cache_hit("slack")
        record_user_mapping_cache_miss("slack")
        set_user_mappings_active("slack", 10)

    def test_metrics_init_idempotent(self):
        """Multiple init calls should be safe."""
        _init_noop_metrics()
        _init_noop_metrics()  # Should not error
        _init_noop_metrics()

        # Metrics should still work
        record_task_queue_operation("test", True, 0.001)
