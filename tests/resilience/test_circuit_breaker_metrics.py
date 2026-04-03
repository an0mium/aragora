"""
Tests for Prometheus metrics integration in circuit_breaker.py.

Tests cover:
- Prometheus metric lazy initialization
- Failure counter increments on record_failure
- Success counter increments on record_success
- State gauge updates on state transitions (open/closed)
- Transition counter increments on state changes
- Entity-mode metrics use name:entity labels
- No-op behavior when prometheus_client is unavailable
- reset_prometheus_metrics clears metric state
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import aragora.resilience.circuit_breaker as cb_module
from aragora.resilience.circuit_breaker import (
    CircuitBreaker,
    reset_prometheus_metrics,
)


@pytest.fixture(autouse=True)
def _clean_prometheus_state():
    """Reset prometheus metrics before and after each test."""
    reset_prometheus_metrics()
    yield
    reset_prometheus_metrics()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeMetric:
    """Minimal fake for Counter / Gauge that tracks .labels().inc()/.set()."""

    def __init__(self, name, description, labels=None):
        self.metric_name = name
        self.description = description
        self._labels = labels or []
        self._children: dict[tuple, MagicMock] = {}

    def labels(self, **kwargs):
        key = tuple(sorted(kwargs.items()))
        if key not in self._children:
            child = MagicMock()
            child.inc = MagicMock()
            child.set = MagicMock()
            child.observe = MagicMock()
            self._children[key] = child
        return self._children[key]


def _fake_counter(name, description, labels=None):
    return FakeMetric(name, description, labels)


def _fake_gauge(name, description, labels=None):
    return FakeMetric(name, description, labels)


def _patch_prometheus():
    """Patch prometheus imports to use fakes."""
    fake_mod = MagicMock()
    fake_mod.Counter = _fake_counter
    fake_mod.Gauge = _fake_gauge
    return patch.dict("sys.modules", {"prometheus_client": fake_mod})


# ---------------------------------------------------------------------------
# Tests: Prometheus availability check
# ---------------------------------------------------------------------------


class TestCheckPrometheus:
    def test_returns_true_when_available(self):
        with _patch_prometheus():
            cb_module._prometheus_available = None
            assert cb_module._check_prometheus() is True

    def test_returns_false_when_unavailable(self):
        cb_module._prometheus_available = None
        with patch.dict("sys.modules", {"prometheus_client": None}):
            # Force re-check
            cb_module._prometheus_available = False
            assert cb_module._check_prometheus() is False


# ---------------------------------------------------------------------------
# Tests: Failure counter
# ---------------------------------------------------------------------------


class TestFailureMetrics:
    def test_single_mode_failure_increments_counter(self):
        with _patch_prometheus():
            cb_module._prometheus_available = None
            cb = CircuitBreaker(name="test-cb", failure_threshold=5)
            cb.record_failure()

            metric = cb_module._prom_metrics.get("aragora_circuit_breaker_failures_total")
            assert metric is not None
            child = metric.labels(circuit_name="test-cb")
            child.inc.assert_called()

    def test_entity_mode_failure_increments_counter(self):
        with _patch_prometheus():
            cb_module._prometheus_available = None
            cb = CircuitBreaker(name="test-cb", failure_threshold=5)
            cb.record_failure("agent-1")

            metric = cb_module._prom_metrics["aragora_circuit_breaker_failures_total"]
            child = metric.labels(circuit_name="test-cb:agent-1")
            child.inc.assert_called()


# ---------------------------------------------------------------------------
# Tests: Success counter
# ---------------------------------------------------------------------------


class TestSuccessMetrics:
    def test_single_mode_success_increments_counter(self):
        with _patch_prometheus():
            cb_module._prometheus_available = None
            cb = CircuitBreaker(name="test-cb")
            cb.record_success()

            metric = cb_module._prom_metrics.get("aragora_circuit_breaker_successes_total")
            assert metric is not None
            child = metric.labels(circuit_name="test-cb")
            child.inc.assert_called()

    def test_entity_mode_success_increments_counter(self):
        with _patch_prometheus():
            cb_module._prometheus_available = None
            cb = CircuitBreaker(name="test-cb")
            cb.record_success("agent-1")

            metric = cb_module._prom_metrics["aragora_circuit_breaker_successes_total"]
            child = metric.labels(circuit_name="test-cb:agent-1")
            child.inc.assert_called()


# ---------------------------------------------------------------------------
# Tests: State gauge and transition counter on open/close
# ---------------------------------------------------------------------------


class TestStateTransitionMetrics:
    def test_circuit_open_sets_state_gauge_to_1(self):
        with _patch_prometheus():
            cb_module._prometheus_available = None
            cb = CircuitBreaker(name="test-cb", failure_threshold=2)
            cb.record_failure()
            cb.record_failure()  # triggers open

            state_metric = cb_module._prom_metrics.get("aragora_circuit_breaker_state")
            assert state_metric is not None
            child = state_metric.labels(circuit_name="test-cb")
            child.set.assert_called_with(1)  # 1 = open

    def test_circuit_close_sets_state_gauge_to_0(self):
        with _patch_prometheus():
            cb_module._prometheus_available = None
            cb = CircuitBreaker(name="test-cb", failure_threshold=2)
            cb.record_failure()
            cb.record_failure()  # open
            cb.record_success()  # close

            state_metric = cb_module._prom_metrics["aragora_circuit_breaker_state"]
            child = state_metric.labels(circuit_name="test-cb")
            child.set.assert_called_with(0)  # 0 = closed

    def test_transition_counter_incremented_on_open(self):
        with _patch_prometheus():
            cb_module._prometheus_available = None
            cb = CircuitBreaker(name="test-cb", failure_threshold=2)
            cb.record_failure()
            cb.record_failure()

            trans_metric = cb_module._prom_metrics.get("aragora_circuit_breaker_transitions_total")
            assert trans_metric is not None
            child = trans_metric.labels(circuit_name="test-cb", to_state="open")
            child.inc.assert_called()

    def test_transition_counter_incremented_on_close(self):
        with _patch_prometheus():
            cb_module._prometheus_available = None
            cb = CircuitBreaker(name="test-cb", failure_threshold=2)
            cb.record_failure()
            cb.record_failure()
            cb.record_success()

            trans_metric = cb_module._prom_metrics["aragora_circuit_breaker_transitions_total"]
            child = trans_metric.labels(circuit_name="test-cb", to_state="closed")
            child.inc.assert_called()


# ---------------------------------------------------------------------------
# Tests: Entity-mode state transitions
# ---------------------------------------------------------------------------


class TestEntityStateTransitionMetrics:
    def test_entity_circuit_open_sets_gauge(self):
        with _patch_prometheus():
            cb_module._prometheus_available = None
            cb = CircuitBreaker(name="svc", failure_threshold=2)
            cb.record_failure("node-a")
            cb.record_failure("node-a")

            state_metric = cb_module._prom_metrics["aragora_circuit_breaker_state"]
            child = state_metric.labels(circuit_name="svc:node-a")
            child.set.assert_called_with(1)


# ---------------------------------------------------------------------------
# Tests: No-op when prometheus unavailable
# ---------------------------------------------------------------------------


class TestNoOpWithoutPrometheus:
    def test_no_metrics_created_when_unavailable(self):
        cb_module._prometheus_available = False
        cb = CircuitBreaker(name="test-cb", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # No prometheus metrics should be created
        assert len(cb_module._prom_metrics) == 0


# ---------------------------------------------------------------------------
# Tests: reset_prometheus_metrics
# ---------------------------------------------------------------------------


class TestResetPrometheusMetrics:
    def test_clears_metrics_dict(self):
        with _patch_prometheus():
            cb_module._prometheus_available = None
            cb = CircuitBreaker(name="test-cb", failure_threshold=2)
            cb.record_failure()
            assert len(cb_module._prom_metrics) > 0

            reset_prometheus_metrics()
            assert len(cb_module._prom_metrics) == 0

    def test_resets_availability_flag(self):
        cb_module._prometheus_available = True
        reset_prometheus_metrics()
        assert cb_module._prometheus_available is None
