"""
Tests for resilience_patterns.metrics module.

Tests cover:
- Circuit breaker state change metrics
- Retry attempt metrics
- Timeout occurrence metrics
- Health status change metrics
- Metrics callback factory
- Graceful degradation when prometheus_client unavailable
"""

import pytest
from unittest.mock import MagicMock, patch

from aragora.resilience_patterns.metrics import (
    circuit_breaker_state_changed,
    retry_attempt,
    retry_exhausted,
    timeout_occurred,
    health_status_changed,
    operation_duration,
    create_metrics_callbacks,
    reset_metrics,
)


class TestMetricsWithoutPrometheus:
    """Tests for graceful degradation when prometheus_client is not available."""

    def test_circuit_breaker_no_prometheus(self):
        """Test circuit_breaker_state_changed works without prometheus."""
        with patch.dict("sys.modules", {"prometheus_client": None}):
            # Reset module state
            import aragora.resilience_patterns.metrics as m

            m._prometheus_available = None
            m._metrics.clear()

            # Should not raise
            circuit_breaker_state_changed("test", "closed", "open")

    def test_retry_no_prometheus(self):
        """Test retry_attempt works without prometheus."""
        with patch.dict("sys.modules", {"prometheus_client": None}):
            import aragora.resilience_patterns.metrics as m

            m._prometheus_available = None
            m._metrics.clear()

            retry_attempt("test_op", 1, 1.0)

    def test_timeout_no_prometheus(self):
        """Test timeout_occurred works without prometheus."""
        with patch.dict("sys.modules", {"prometheus_client": None}):
            import aragora.resilience_patterns.metrics as m

            m._prometheus_available = None
            m._metrics.clear()

            timeout_occurred("test_op", 5.0)


class TestCircuitBreakerMetrics:
    """Tests for circuit breaker metrics."""

    def setup_method(self):
        """Reset metrics before each test."""
        reset_metrics()

    def test_state_change_records_metric(self):
        """Test that state changes are recorded."""
        # Should not raise
        circuit_breaker_state_changed("my_breaker", "closed", "open")
        circuit_breaker_state_changed("my_breaker", "open", "half_open")
        circuit_breaker_state_changed("my_breaker", "half_open", "closed")

    def test_state_change_with_enum_values(self):
        """Test state changes with enum-like objects."""
        from aragora.resilience_patterns import CircuitState

        circuit_breaker_state_changed("test_breaker", CircuitState.CLOSED, CircuitState.OPEN)

    def test_callback_signature_matches_config(self):
        """Test callback signature matches CircuitBreakerConfig.on_state_change."""
        from aragora.resilience_patterns import CircuitBreakerConfig, CircuitState

        # The callback should accept (name, old_state, new_state)
        config = CircuitBreakerConfig(on_state_change=circuit_breaker_state_changed)

        # Simulate calling the callback as CircuitBreaker would
        config.on_state_change("test", CircuitState.CLOSED, CircuitState.OPEN)


class TestRetryMetrics:
    """Tests for retry metrics."""

    def setup_method(self):
        """Reset metrics before each test."""
        reset_metrics()

    def test_retry_attempt_records_metric(self):
        """Test that retry attempts are recorded."""
        retry_attempt("my_operation", 1, 0.5)
        retry_attempt("my_operation", 2, 1.0)
        retry_attempt("my_operation", 3, 2.0)

    def test_retry_with_exception(self):
        """Test retry with exception info."""
        error = ValueError("test error")
        retry_attempt("failing_op", 1, 0.5, exception=error)

    def test_retry_exhausted_records_metric(self):
        """Test that exhausted retries are recorded."""
        retry_exhausted("my_operation", 3)
        retry_exhausted("my_operation", 3, last_exception=ValueError("final error"))


class TestTimeoutMetrics:
    """Tests for timeout metrics."""

    def setup_method(self):
        """Reset metrics before each test."""
        reset_metrics()

    def test_timeout_occurred_records_metric(self):
        """Test that timeouts are recorded."""
        timeout_occurred("slow_operation", 5.0)
        timeout_occurred("slow_operation", 10.0)

    def test_timeout_with_various_values(self):
        """Test timeout with various timeout values."""
        timeout_occurred("op_1", 0.1)
        timeout_occurred("op_2", 30.0)
        timeout_occurred("op_3", 180.0)


class TestHealthMetrics:
    """Tests for health status metrics."""

    def setup_method(self):
        """Reset metrics before each test."""
        reset_metrics()

    def test_health_status_healthy(self):
        """Test recording healthy status."""
        health_status_changed("database", True)

    def test_health_status_unhealthy(self):
        """Test recording unhealthy status."""
        health_status_changed("cache", False, consecutive_failures=3)

    def test_health_status_transitions(self):
        """Test health status transitions."""
        health_status_changed("service", True)
        health_status_changed("service", False, consecutive_failures=1)
        health_status_changed("service", False, consecutive_failures=2)
        health_status_changed("service", True, consecutive_failures=0)


class TestOperationDuration:
    """Tests for operation duration metrics."""

    def setup_method(self):
        """Reset metrics before each test."""
        reset_metrics()

    def test_successful_operation(self):
        """Test recording successful operation duration."""
        operation_duration("api_call", 0.5, success=True)

    def test_failed_operation(self):
        """Test recording failed operation duration."""
        operation_duration("api_call", 2.0, success=False)

    def test_multiple_operations(self):
        """Test recording multiple operation durations."""
        for i in range(10):
            operation_duration("batch_op", 0.1 * i, success=i % 2 == 0)


class TestMetricsCallbackFactory:
    """Tests for create_metrics_callbacks factory function."""

    def test_returns_all_callbacks(self):
        """Test that factory returns all expected callbacks."""
        callbacks = create_metrics_callbacks()

        assert "on_circuit_state_change" in callbacks
        assert "on_retry" in callbacks
        assert "on_retry_exhausted" in callbacks
        assert "on_timeout" in callbacks
        assert "on_health_change" in callbacks
        assert "on_operation_complete" in callbacks

    def test_callbacks_are_callable(self):
        """Test that all returned callbacks are callable."""
        callbacks = create_metrics_callbacks()

        for name, callback in callbacks.items():
            assert callable(callback), f"{name} is not callable"

    def test_circuit_callback_works(self):
        """Test circuit state change callback from factory."""
        callbacks = create_metrics_callbacks()
        callback = callbacks["on_circuit_state_change"]

        # Should not raise
        callback("test_breaker", "closed", "open")

    def test_retry_callback_works(self):
        """Test retry callback from factory."""
        callbacks = create_metrics_callbacks()
        callback = callbacks["on_retry"]

        callback("test_op", 1, 0.5)

    def test_timeout_callback_works(self):
        """Test timeout callback from factory."""
        callbacks = create_metrics_callbacks()
        callback = callbacks["on_timeout"]

        callback("test_op", 5.0)


class TestResetMetrics:
    """Tests for reset_metrics function."""

    def test_reset_clears_metrics(self):
        """Test that reset clears all metrics."""
        # Create some metrics
        circuit_breaker_state_changed("test", "closed", "open")
        retry_attempt("test", 1, 0.5)
        timeout_occurred("test", 5.0)

        # Reset
        reset_metrics()

        # Metrics should be cleared (no way to verify without prometheus,
        # but we verify the function runs without error)


class TestMetricsIntegration:
    """Integration tests for metrics with resilience patterns."""

    def setup_method(self):
        """Reset metrics before each test."""
        reset_metrics()

    def test_circuit_breaker_with_metrics_callback(self):
        """Test circuit breaker with metrics callback."""
        from aragora.resilience_patterns import (
            BaseCircuitBreaker,
            CircuitBreakerConfig,
        )

        config = CircuitBreakerConfig(
            failure_threshold=2,
            on_state_change=circuit_breaker_state_changed,
        )

        cb = BaseCircuitBreaker("test_service", config)

        # Trigger state changes
        cb.record_failure()
        cb.record_failure()  # Should open

        assert cb.is_open

    def test_health_checker_with_metrics(self):
        """Test health checker reporting to metrics."""
        from aragora.resilience_patterns import HealthChecker

        checker = HealthChecker("database", failure_threshold=2)

        # Record some activity
        checker.record_success(latency_ms=50.0)
        status = checker.get_status()

        # Report to metrics
        health_status_changed(
            "database",
            status.healthy,
            status.consecutive_failures,
        )
