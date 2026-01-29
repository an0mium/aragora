"""
Tests for aragora.observability.metrics module.

Covers:
- NoOpMetric fallback behavior
- Metric initialization (noop when disabled)
- Recording functions (request, agent call, debate, memory, etc.)
- Endpoint normalization
- Context managers (track_debate, track_phase, track_bridge_sync)
- Decorators (measure_latency, measure_async_latency)
- Cross-functional metric recording
- KM metric recording
- Notification metric recording
- Task queue metric recording
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.observability.config import MetricsConfig, set_metrics_config, reset_config
from aragora.observability.metrics.base import NoOpMetric


# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_metrics_state():
    """Reset metrics module state between tests."""
    import aragora.observability.metrics as m

    m._initialized = False
    m._metrics_server = None
    # Disable metrics to avoid requiring prometheus_client in tests
    set_metrics_config(MetricsConfig(enabled=False))
    yield
    reset_config()
    m._initialized = False
    m._metrics_server = None


# =============================================================================
# TestNoOpMetric
# =============================================================================


class TestNoOpMetric:
    """Tests for NoOpMetric fallback class."""

    def test_labels_returns_self(self):
        """labels() should return self for chaining."""
        m = NoOpMetric()
        result = m.labels(foo="bar")
        assert result is m

    def test_labels_chaining(self):
        """Multiple labels() calls should chain."""
        m = NoOpMetric()
        result = m.labels(a="1").labels(b="2")
        assert result is m

    def test_inc_no_error(self):
        """inc() should not raise."""
        NoOpMetric().inc()
        NoOpMetric().inc(5)

    def test_dec_no_error(self):
        """dec() should not raise."""
        NoOpMetric().dec()
        NoOpMetric().dec(3)

    def test_set_no_error(self):
        """set() should not raise."""
        NoOpMetric().set(42.0)

    def test_observe_no_error(self):
        """observe() should not raise."""
        NoOpMetric().observe(1.5)

    def test_chained_operations(self):
        """Chained labels + inc should not raise."""
        NoOpMetric().labels(agent="claude", status="ok").inc()

    def test_chained_observe(self):
        """Chained labels + observe should not raise."""
        NoOpMetric().labels(endpoint="/api/test").observe(0.5)


# =============================================================================
# TestMetricsInitialization
# =============================================================================


class TestMetricsInitialization:
    """Tests for metrics initialization."""

    def test_init_with_disabled_config(self):
        """Should initialize noop metrics when disabled."""
        import aragora.observability.metrics as m

        m._init_metrics()

        assert m._initialized is True
        # Globals should be NoOpMetric instances
        assert isinstance(m.REQUEST_COUNT, NoOpMetric)
        assert isinstance(m.AGENT_CALLS, NoOpMetric)
        assert isinstance(m.ACTIVE_DEBATES, NoOpMetric)
        assert isinstance(m.MEMORY_OPERATIONS, NoOpMetric)

    def test_init_idempotent(self):
        """Second init call should return immediately."""
        import aragora.observability.metrics as m

        m._init_metrics()
        first_count = m.REQUEST_COUNT
        m._init_metrics()
        assert m.REQUEST_COUNT is first_count

    def test_init_returns_false_when_disabled(self):
        """Should return False when metrics are disabled."""
        import aragora.observability.metrics as m

        result = m._init_metrics()
        assert result is False

    def test_init_core_metrics_alias(self):
        """init_core_metrics should be alias for _init_metrics."""
        import aragora.observability.metrics as m

        result = m.init_core_metrics()
        assert m._initialized is True
        assert result is False  # disabled

    def test_noop_debate_metrics_initialized(self):
        """Debate-specific noop metrics should be set."""
        import aragora.observability.metrics as m

        m._init_metrics()
        assert isinstance(m.DEBATE_DURATION, NoOpMetric)
        assert isinstance(m.DEBATE_ROUNDS, NoOpMetric)
        assert isinstance(m.DEBATE_PHASE_DURATION, NoOpMetric)
        assert isinstance(m.AGENT_PARTICIPATION, NoOpMetric)

    def test_noop_cache_metrics_initialized(self):
        """Cache noop metrics should be set."""
        import aragora.observability.metrics as m

        m._init_metrics()
        assert isinstance(m.CACHE_HITS, NoOpMetric)
        assert isinstance(m.CACHE_MISSES, NoOpMetric)

    def test_noop_cross_functional_metrics(self):
        """Cross-functional noop metrics should be set."""
        import aragora.observability.metrics as m

        m._init_metrics()
        assert isinstance(m.KNOWLEDGE_CACHE_HITS, NoOpMetric)
        assert isinstance(m.MEMORY_COORDINATOR_WRITES, NoOpMetric)
        assert isinstance(m.WORKFLOW_TRIGGERS, NoOpMetric)
        assert isinstance(m.EVIDENCE_STORED, NoOpMetric)


# =============================================================================
# TestRecordingFunctions
# =============================================================================


class TestRecordingFunctions:
    """Tests for metric recording functions (with noop metrics)."""

    def test_record_request(self):
        """record_request should not raise with noop metrics."""
        from aragora.observability.metrics import record_request

        record_request("GET", "/api/debates", 200, 0.05)
        record_request("POST", "/api/debates", 201, 0.12)
        record_request("GET", "/api/debates", 500, 1.5)

    def test_record_agent_call(self):
        """record_agent_call should not raise with noop metrics."""
        from aragora.observability.metrics import record_agent_call

        record_agent_call("claude", success=True, latency=1.2)
        record_agent_call("gpt-4", success=False, latency=30.0)

    def test_set_consensus_rate(self):
        """set_consensus_rate should not raise."""
        from aragora.observability.metrics import set_consensus_rate

        set_consensus_rate(0.85)
        set_consensus_rate(0.0)
        set_consensus_rate(1.0)

    def test_record_memory_operation(self):
        """record_memory_operation should not raise."""
        from aragora.observability.metrics import record_memory_operation

        record_memory_operation("store", "fast")
        record_memory_operation("query", "slow")
        record_memory_operation("promote", "glacial")

    def test_track_websocket_connection(self):
        """track_websocket_connection should not raise."""
        from aragora.observability.metrics import track_websocket_connection

        track_websocket_connection(True)
        track_websocket_connection(False)

    def test_record_debate_completion(self):
        """record_debate_completion should not raise."""
        from aragora.observability.metrics import record_debate_completion

        record_debate_completion(120.5, 3, "consensus")
        record_debate_completion(60.0, 5, "no_consensus")
        record_debate_completion(5.0, 1, "error")

    def test_record_phase_duration(self):
        """record_phase_duration should not raise."""
        from aragora.observability.metrics import record_phase_duration

        record_phase_duration("propose", 10.5)
        record_phase_duration("critique", 8.2)
        record_phase_duration("vote", 3.0)

    def test_record_agent_participation(self):
        """record_agent_participation should not raise."""
        from aragora.observability.metrics import record_agent_participation

        record_agent_participation("claude", "propose")
        record_agent_participation("gpt-4", "critique")

    def test_record_cache_hit_miss(self):
        """Cache hit/miss recording should not raise."""
        from aragora.observability.metrics import record_cache_hit, record_cache_miss

        record_cache_hit("knowledge")
        record_cache_miss("knowledge")
        record_cache_hit("elo")


# =============================================================================
# TestEndpointNormalization
# =============================================================================


class TestEndpointNormalization:
    """Tests for _normalize_endpoint."""

    def test_replaces_uuid(self):
        """Should replace UUIDs with :id."""
        from aragora.observability.metrics import _normalize_endpoint

        result = _normalize_endpoint("/api/debates/550e8400-e29b-41d4-a716-446655440000")
        assert ":id" in result
        assert "550e8400" not in result

    def test_replaces_numeric_ids(self):
        """Should replace numeric path segments with :id."""
        from aragora.observability.metrics import _normalize_endpoint

        result = _normalize_endpoint("/api/users/12345/posts")
        assert "/:id/" in result
        assert "12345" not in result

    def test_replaces_long_tokens(self):
        """Should replace long base64-like tokens."""
        from aragora.observability.metrics import _normalize_endpoint

        result = _normalize_endpoint("/api/verify/abcdefghijklmnopqrstuvwxyz")
        assert ":token" in result

    def test_preserves_static_paths(self):
        """Should preserve static path segments."""
        from aragora.observability.metrics import _normalize_endpoint

        result = _normalize_endpoint("/api/debates")
        assert result == "/api/debates"

    def test_handles_empty_path(self):
        """Should handle empty/root paths."""
        from aragora.observability.metrics import _normalize_endpoint

        result = _normalize_endpoint("/")
        assert result == "/"


# =============================================================================
# TestContextManagers
# =============================================================================


class TestContextManagers:
    """Tests for metric context managers."""

    def test_track_debate(self):
        """track_debate should inc/dec ACTIVE_DEBATES."""
        from aragora.observability.metrics import track_debate

        with track_debate():
            pass  # No error

    def test_track_debate_on_exception(self):
        """track_debate should still dec on exception."""
        from aragora.observability.metrics import track_debate

        with pytest.raises(ValueError):
            with track_debate():
                raise ValueError("test error")

    def test_track_phase(self):
        """track_phase should measure phase duration."""
        from aragora.observability.metrics import track_phase

        with track_phase("propose"):
            time.sleep(0.001)

    def test_track_bridge_sync(self):
        """track_bridge_sync should record sync metrics."""
        from aragora.observability.metrics import track_bridge_sync

        with track_bridge_sync("evidence"):
            pass

    def test_track_bridge_sync_failure(self):
        """track_bridge_sync should record failure on exception."""
        from aragora.observability.metrics import track_bridge_sync

        with pytest.raises(RuntimeError):
            with track_bridge_sync("evidence"):
                raise RuntimeError("sync failed")


# =============================================================================
# TestDecorators
# =============================================================================


class TestDecorators:
    """Tests for measure_latency decorators."""

    def test_measure_latency_sync(self):
        """measure_latency should wrap sync function."""
        from aragora.observability.metrics import measure_latency

        @measure_latency("test_endpoint")
        def my_func(x):
            return x * 2

        result = my_func(5)
        assert result == 10

    def test_measure_async_latency(self):
        """measure_async_latency should wrap async function."""
        from aragora.observability.metrics import measure_async_latency

        @measure_async_latency("test_endpoint")
        async def my_func(x):
            return x * 2

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(my_func(5))
        assert result == 10


# =============================================================================
# TestCrossFunctionalMetrics
# =============================================================================


class TestCrossFunctionalMetrics:
    """Tests for cross-functional metric recording."""

    def test_record_knowledge_cache(self):
        """Knowledge cache recording should not raise."""
        from aragora.observability.metrics import (
            record_knowledge_cache_hit,
            record_knowledge_cache_miss,
        )

        record_knowledge_cache_hit()
        record_knowledge_cache_miss()

    def test_record_memory_coordinator_write(self):
        """Memory coordinator write recording should not raise."""
        from aragora.observability.metrics import record_memory_coordinator_write

        record_memory_coordinator_write("success")
        record_memory_coordinator_write("partial")

    def test_record_selection_feedback_adjustment(self):
        """Selection feedback adjustment should not raise."""
        from aragora.observability.metrics import record_selection_feedback_adjustment

        record_selection_feedback_adjustment("claude", "up")
        record_selection_feedback_adjustment("gpt-4", "down")

    def test_record_workflow_trigger(self):
        """Workflow trigger recording should not raise."""
        from aragora.observability.metrics import record_workflow_trigger

        record_workflow_trigger("success")
        record_workflow_trigger("failure")

    def test_record_evidence_stored(self):
        """Evidence stored recording should not raise."""
        from aragora.observability.metrics import record_evidence_stored

        record_evidence_stored()
        record_evidence_stored(5)

    def test_record_culture_patterns(self):
        """Culture pattern recording should not raise."""
        from aragora.observability.metrics import record_culture_patterns

        record_culture_patterns()
        record_culture_patterns(3)


# =============================================================================
# TestKMMetrics
# =============================================================================


class TestKMMetrics:
    """Tests for Knowledge Mound metric recording."""

    def test_record_km_operation(self):
        """KM operation recording should not raise."""
        from aragora.observability.metrics import record_km_operation

        record_km_operation("store", success=True, latency_seconds=0.05)
        record_km_operation("query", success=False, latency_seconds=1.2)

    def test_record_km_cache_access(self):
        """KM cache access recording should not raise."""
        from aragora.observability.metrics import record_km_cache_access

        record_km_cache_access(hit=True, adapter="evidence")
        record_km_cache_access(hit=False, adapter="global")

    def test_set_km_health_status(self):
        """KM health status setting should not raise."""
        from aragora.observability.metrics import set_km_health_status

        set_km_health_status(1)
        set_km_health_status(0)

    def test_record_km_adapter_sync(self):
        """KM adapter sync recording should not raise."""
        from aragora.observability.metrics import record_km_adapter_sync

        record_km_adapter_sync("evidence", "inbound", success=True)
        record_km_adapter_sync("elo", "outbound", success=False)

    def test_record_km_federated_query(self):
        """KM federated query recording should not raise."""
        from aragora.observability.metrics import record_km_federated_query

        record_km_federated_query(adapters_queried=5, success=True)

    def test_record_km_event_emitted(self):
        """KM event emitted recording should not raise."""
        from aragora.observability.metrics import record_km_event_emitted

        record_km_event_emitted("store")
        record_km_event_emitted("invalidate")

    def test_set_km_active_adapters(self):
        """KM active adapters setting should not raise."""
        from aragora.observability.metrics import set_km_active_adapters

        set_km_active_adapters(14)


# =============================================================================
# TestSlowDebateMetrics
# =============================================================================


class TestSlowDebateMetrics:
    """Tests for slow debate detection metrics."""

    def test_record_slow_debate(self):
        """Slow debate recording should not raise."""
        from aragora.observability.metrics import record_slow_debate

        record_slow_debate()

    def test_record_slow_round(self):
        """Slow round recording should not raise."""
        from aragora.observability.metrics import record_slow_round

        record_slow_round()
        record_slow_round("consensus")

    def test_record_round_latency(self):
        """Round latency recording should not raise."""
        from aragora.observability.metrics import record_round_latency

        record_round_latency(5.0)
        record_round_latency(0.001)


# =============================================================================
# TestStartMetricsServer
# =============================================================================


class TestStartMetricsServer:
    """Tests for metrics server startup."""

    def test_returns_none_when_disabled(self):
        """Should return None when metrics are disabled."""
        from aragora.observability.metrics import start_metrics_server

        result = start_metrics_server()
        assert result is None
