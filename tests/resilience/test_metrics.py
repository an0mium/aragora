"""
Tests for circuit breaker metrics and status functions.

Tests cover:
- set_metrics_callback
- emit_metrics
- get_circuit_breaker_status
- get_circuit_breaker_metrics
- get_all_circuit_breakers_status
- get_circuit_breaker_summary
"""

from __future__ import annotations

import time

import pytest

from aragora.resilience.circuit_breaker import CircuitBreaker
from aragora.resilience.metrics import (
    emit_metrics,
    get_all_circuit_breakers_status,
    get_circuit_breaker_metrics,
    get_circuit_breaker_status,
    get_circuit_breaker_summary,
    set_metrics_callback,
)
from aragora.resilience.registry import (
    _circuit_breakers,
    _circuit_breakers_lock,
    get_circuit_breaker,
)


@pytest.fixture(autouse=True)
def clean_state():
    """Clean registry and metrics callback before/after each test."""
    with _circuit_breakers_lock:
        _circuit_breakers.clear()
    set_metrics_callback(None)
    yield
    with _circuit_breakers_lock:
        _circuit_breakers.clear()
    set_metrics_callback(None)


# ============================================================================
# set_metrics_callback / emit_metrics Tests
# ============================================================================


class TestMetricsCallback:
    """Tests for metrics callback."""

    def test_set_callback(self):
        """Test setting a metrics callback."""
        calls = []
        set_metrics_callback(lambda name, state: calls.append((name, state)))
        emit_metrics("test", 1)
        assert calls == [("test", 1)]

    def test_clear_callback(self):
        """Test clearing the metrics callback."""
        calls = []
        set_metrics_callback(lambda name, state: calls.append((name, state)))
        set_metrics_callback(None)
        emit_metrics("test", 1)
        assert calls == []

    def test_callback_error_handled(self):
        """Test errors in callback are handled gracefully."""
        set_metrics_callback(lambda name, state: 1 / 0)
        # Should not raise
        emit_metrics("test", 0)

    def test_no_callback_no_error(self):
        """Test emit_metrics with no callback does nothing."""
        emit_metrics("test", 0)  # Should not raise


# ============================================================================
# get_circuit_breaker_status Tests
# ============================================================================


class TestGetCircuitBreakerStatus:
    """Tests for get_circuit_breaker_status."""

    def test_empty_registry(self):
        """Test status with empty registry."""
        status = get_circuit_breaker_status()
        assert status["_registry_size"] == 0

    def test_single_closed_circuit(self):
        """Test status with one closed circuit breaker."""
        get_circuit_breaker("test-svc")
        status = get_circuit_breaker_status()
        assert status["_registry_size"] == 1
        assert "test-svc" in status
        assert status["test-svc"]["status"] == "closed"
        assert status["test-svc"]["failures"] == 0

    def test_open_circuit(self):
        """Test status with an open circuit breaker."""
        cb = get_circuit_breaker("failing-svc", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()

        status = get_circuit_breaker_status()
        assert status["failing-svc"]["status"] == "open"


# ============================================================================
# get_circuit_breaker_metrics Tests
# ============================================================================


class TestGetCircuitBreakerMetrics:
    """Tests for get_circuit_breaker_metrics."""

    def test_empty_registry_metrics(self):
        """Test metrics with empty registry."""
        metrics = get_circuit_breaker_metrics()
        assert metrics["registry_size"] == 0
        assert metrics["summary"]["total"] == 0
        assert metrics["health"]["status"] == "healthy"

    def test_healthy_metrics(self):
        """Test metrics with all circuits healthy."""
        get_circuit_breaker("svc-1")
        get_circuit_breaker("svc-2")

        metrics = get_circuit_breaker_metrics()
        assert metrics["summary"]["total"] == 2
        assert metrics["summary"]["closed"] == 2
        assert metrics["summary"]["open"] == 0
        assert metrics["health"]["status"] == "healthy"

    def test_degraded_health(self):
        """Test metrics show degraded when circuit is open."""
        cb = get_circuit_breaker("bad-svc", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()

        metrics = get_circuit_breaker_metrics()
        assert metrics["summary"]["open"] == 1
        assert metrics["health"]["status"] == "degraded"
        assert "bad-svc" in metrics["health"]["open_circuits"]

    def test_critical_health(self):
        """Test metrics show critical when 3+ circuits are open."""
        for i in range(3):
            cb = get_circuit_breaker(f"bad-{i}", failure_threshold=2)
            cb.record_failure()
            cb.record_failure()

        metrics = get_circuit_breaker_metrics()
        assert metrics["summary"]["open"] == 3
        assert metrics["health"]["status"] == "critical"

    def test_high_failure_circuits(self):
        """Test high-failure circuit detection."""
        cb = get_circuit_breaker("warn-svc", failure_threshold=10)
        for _ in range(6):
            cb.record_failure()

        metrics = get_circuit_breaker_metrics()
        high_failure = metrics["health"]["high_failure_circuits"]
        assert len(high_failure) == 1
        assert high_failure[0]["name"] == "warn-svc"
        assert high_failure[0]["percentage"] == 60.0

    def test_per_circuit_metrics(self):
        """Test per-circuit-breaker metrics details."""
        cb = get_circuit_breaker("detail-svc", failure_threshold=5)
        cb.record_failure()

        metrics = get_circuit_breaker_metrics()
        detail = metrics["circuit_breakers"]["detail-svc"]
        assert detail["status"] == "closed"
        assert detail["failures"] == 1
        assert detail["failure_threshold"] == 5
        assert "cooldown_seconds" in detail
        assert "entity_mode" in detail


# ============================================================================
# get_all_circuit_breakers_status Tests
# ============================================================================


class TestGetAllStatus:
    """Tests for get_all_circuit_breakers_status."""

    def test_empty_registry(self):
        """Test status with empty registry."""
        status = get_all_circuit_breakers_status()
        assert status["healthy"] is True
        assert status["total_circuits"] == 0

    def test_all_closed(self):
        """Test all circuits closed."""
        get_circuit_breaker("svc-1")
        get_circuit_breaker("svc-2")

        status = get_all_circuit_breakers_status()
        assert status["healthy"] is True
        assert status["total_circuits"] == 2
        assert status["closed_circuits"] == 2
        assert status["open_circuits"] == 0

    def test_with_open_circuit(self):
        """Test with open circuit."""
        cb = get_circuit_breaker("fail-svc", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()

        status = get_all_circuit_breakers_status()
        assert status["healthy"] is False
        assert status["open_circuits"] == 1
        assert "fail-svc" in status["circuits"]
        assert status["circuits"]["fail-svc"]["status"] == "open"


# ============================================================================
# get_circuit_breaker_summary Tests
# ============================================================================


class TestGetSummary:
    """Tests for get_circuit_breaker_summary."""

    def test_empty_registry(self):
        """Test summary with empty registry."""
        summary = get_circuit_breaker_summary()
        assert summary["healthy"] is True
        assert summary["total"] == 0
        assert summary["open"] == []
        assert summary["half_open"] == []

    def test_healthy_summary(self):
        """Test summary when all circuits are healthy."""
        get_circuit_breaker("svc-1")
        get_circuit_breaker("svc-2")

        summary = get_circuit_breaker_summary()
        assert summary["healthy"] is True
        assert summary["total"] == 2

    def test_unhealthy_summary(self):
        """Test summary with open circuits."""
        cb = get_circuit_breaker("down-svc", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()

        summary = get_circuit_breaker_summary()
        assert summary["healthy"] is False
        assert "down-svc" in summary["open"]

    def test_half_open_in_summary(self):
        """Test half-open circuits appear in summary."""
        cb = get_circuit_breaker("recovering-svc", failure_threshold=2)
        cb.cooldown_seconds = 0.01
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)

        summary = get_circuit_breaker_summary()
        assert "recovering-svc" in summary["half_open"]
