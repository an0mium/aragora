"""
Tests for Prometheus metrics for resilience patterns.

Tests cover:
- _check_prometheus() with and without prometheus_client available
- _get_or_create_metric() lazy creation and caching
- circuit_breaker_state_changed() with enums and strings
- retry_attempt() Counter + Histogram recording
- retry_exhausted() Counter recording
- timeout_occurred() Counter + Histogram recording
- health_status_changed() Gauge recording (healthy/unhealthy)
- operation_duration() Histogram recording with success label
- create_metrics_callbacks() returns expected dict
- reset_metrics() clears the _metrics dict
- All functions are no-ops when prometheus_client is unavailable
"""

from __future__ import annotations

from enum import Enum
from unittest.mock import MagicMock, patch

import pytest

import aragora.resilience.pattern_metrics as pm_module
from aragora.resilience.pattern_metrics import (
    _check_prometheus,
    _get_or_create_metric,
    circuit_breaker_state_changed,
    create_metrics_callbacks,
    health_status_changed,
    operation_duration,
    reset_metrics,
    retry_attempt,
    retry_exhausted,
    timeout_occurred,
)


@pytest.fixture(autouse=True)
def clean_state():
    """Reset module globals before and after each test."""
    pm_module._metrics.clear()
    pm_module._prometheus_available = None
    yield
    pm_module._metrics.clear()
    pm_module._prometheus_available = None


# ============================================================================
# Helper: Mock prometheus metric classes
# ============================================================================


def _make_mock_metric_class(name: str = "MockMetric") -> MagicMock:
    """Create a mock metric class that returns labeled children with inc/set/observe."""

    class FakeMetric:
        def __init__(self, metric_name, description, labels=None):
            self.metric_name = metric_name
            self.description = description
            self._labels = labels or []
            self._children = {}

        def labels(self, **kwargs):
            key = tuple(sorted(kwargs.items()))
            if key not in self._children:
                child = MagicMock()
                child.inc = MagicMock()
                child.set = MagicMock()
                child.observe = MagicMock()
                self._children[key] = child
            return self._children[key]

    FakeMetric.__name__ = name
    return FakeMetric


# ============================================================================
# _check_prometheus Tests
# ============================================================================


class TestCheckPrometheus:
    """Tests for _check_prometheus detection and caching."""

    def test_returns_false_when_import_fails(self):
        """Test that _check_prometheus returns False when prometheus_client is missing."""
        with patch.dict("sys.modules", {"prometheus_client": None}):
            pm_module._prometheus_available = None
            # Importing None module will trigger ImportError-like behavior.
            # Instead, directly simulate via builtins __import__.
            original_import = (
                __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
            )

            def mock_import(name, *args, **kwargs):
                if name == "prometheus_client":
                    raise ImportError("mocked")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                result = _check_prometheus()
                assert result is False
                assert pm_module._prometheus_available is False

    def test_returns_true_when_import_succeeds(self):
        """Test that _check_prometheus returns True when prometheus_client is available."""
        mock_prometheus = MagicMock()
        with patch.dict("sys.modules", {"prometheus_client": mock_prometheus}):
            pm_module._prometheus_available = None
            result = _check_prometheus()
            assert result is True
            assert pm_module._prometheus_available is True

    def test_caches_result_false(self):
        """Test that _check_prometheus uses cached False result."""
        pm_module._prometheus_available = False
        result = _check_prometheus()
        assert result is False

    def test_caches_result_true(self):
        """Test that _check_prometheus uses cached True result."""
        pm_module._prometheus_available = True
        result = _check_prometheus()
        assert result is True


# ============================================================================
# No-op Tests (prometheus_client unavailable)
# ============================================================================


class TestNoOpWhenPrometheusUnavailable:
    """All metric functions should be silent no-ops when prometheus is missing."""

    @pytest.fixture(autouse=True)
    def disable_prometheus(self):
        """Disable prometheus for all tests in this class."""
        pm_module._prometheus_available = False
        yield

    def test_circuit_breaker_state_changed_noop(self):
        """circuit_breaker_state_changed does nothing without prometheus."""
        circuit_breaker_state_changed("test-cb", "closed", "open")
        assert len(pm_module._metrics) == 0

    def test_retry_attempt_noop(self):
        """retry_attempt does nothing without prometheus."""
        retry_attempt("test-op", attempt=1, delay=0.5, exception=RuntimeError("x"))
        assert len(pm_module._metrics) == 0

    def test_retry_exhausted_noop(self):
        """retry_exhausted does nothing without prometheus."""
        retry_exhausted("test-op", total_attempts=3, last_exception=RuntimeError("x"))
        assert len(pm_module._metrics) == 0

    def test_timeout_occurred_noop(self):
        """timeout_occurred does nothing without prometheus."""
        timeout_occurred("test-op", timeout_seconds=30.0)
        assert len(pm_module._metrics) == 0

    def test_health_status_changed_noop(self):
        """health_status_changed does nothing without prometheus."""
        health_status_changed("db", healthy=True, consecutive_failures=0)
        assert len(pm_module._metrics) == 0

    def test_operation_duration_noop(self):
        """operation_duration does nothing without prometheus."""
        operation_duration("test-op", duration_seconds=1.5, success=True)
        assert len(pm_module._metrics) == 0

    def test_get_or_create_metric_returns_none(self):
        """_get_or_create_metric returns None without prometheus."""
        result = _get_or_create_metric("some_metric", MagicMock, "desc", ["label"])
        assert result is None
        assert len(pm_module._metrics) == 0


# ============================================================================
# _get_or_create_metric Tests
# ============================================================================


class TestGetOrCreateMetric:
    """Tests for metric lazy creation and caching."""

    @pytest.fixture(autouse=True)
    def enable_prometheus(self):
        """Enable prometheus for tests in this class."""
        pm_module._prometheus_available = True
        yield

    def test_creates_metric_with_labels(self):
        """Test metric creation with labels."""
        FakeCounter = _make_mock_metric_class("Counter")
        metric = _get_or_create_metric("test_counter", FakeCounter, "A test counter", ["label_a"])
        assert metric is not None
        assert metric.metric_name == "test_counter"
        assert metric.description == "A test counter"
        assert "test_counter" in pm_module._metrics

    def test_creates_metric_without_labels(self):
        """Test metric creation without labels."""
        FakeCounter = _make_mock_metric_class("Counter")
        metric = _get_or_create_metric("test_no_labels", FakeCounter, "No labels")
        assert metric is not None
        assert metric.metric_name == "test_no_labels"

    def test_caches_metric(self):
        """Test that same name returns the same metric instance."""
        FakeCounter = _make_mock_metric_class("Counter")
        first = _get_or_create_metric("cached_metric", FakeCounter, "desc", ["l"])
        second = _get_or_create_metric("cached_metric", FakeCounter, "desc", ["l"])
        assert first is second

    def test_different_names_different_metrics(self):
        """Test that different names create different metrics."""
        FakeCounter = _make_mock_metric_class("Counter")
        m1 = _get_or_create_metric("metric_a", FakeCounter, "desc a", ["l"])
        m2 = _get_or_create_metric("metric_b", FakeCounter, "desc b", ["l"])
        assert m1 is not m2


# ============================================================================
# circuit_breaker_state_changed Tests
# ============================================================================


class TestCircuitBreakerStateChanged:
    """Tests for circuit_breaker_state_changed metric recording."""

    @pytest.fixture(autouse=True)
    def enable_prometheus(self):
        """Enable prometheus and patch Counter import."""
        pm_module._prometheus_available = True
        self.FakeCounter = _make_mock_metric_class("Counter")
        yield

    def test_records_string_states(self):
        """Test recording state change with plain strings."""
        with patch(
            "aragora.resilience.pattern_metrics.Counter",
            self.FakeCounter,
            create=True,
        ):
            # Patch the import inside the function
            import sys

            mock_prom = MagicMock()
            mock_prom.Counter = self.FakeCounter
            with patch.dict(sys.modules, {"prometheus_client": mock_prom}):
                circuit_breaker_state_changed("api-cb", "closed", "open")

        metric = pm_module._metrics.get("resilience_circuit_breaker_state_changes_total")
        assert metric is not None
        key = (
            ("breaker_name", "api-cb"),
            ("from_state", "closed"),
            ("to_state", "open"),
        )
        child = metric._children.get(key)
        assert child is not None
        child.inc.assert_called_once()

    def test_records_enum_states(self):
        """Test recording state change with enum values having .value attribute."""

        class CircuitState(Enum):
            CLOSED = "closed"
            OPEN = "open"
            HALF_OPEN = "half_open"

        import sys

        mock_prom = MagicMock()
        mock_prom.Counter = self.FakeCounter
        with patch.dict(sys.modules, {"prometheus_client": mock_prom}):
            circuit_breaker_state_changed("db-cb", CircuitState.CLOSED, CircuitState.OPEN)

        metric = pm_module._metrics.get("resilience_circuit_breaker_state_changes_total")
        assert metric is not None
        key = (
            ("breaker_name", "db-cb"),
            ("from_state", "closed"),
            ("to_state", "open"),
        )
        child = metric._children.get(key)
        assert child is not None
        child.inc.assert_called_once()

    def test_handles_mixed_enum_and_string(self):
        """Test with one enum and one string state."""

        class CircuitState(Enum):
            CLOSED = "closed"

        import sys

        mock_prom = MagicMock()
        mock_prom.Counter = self.FakeCounter
        with patch.dict(sys.modules, {"prometheus_client": mock_prom}):
            circuit_breaker_state_changed("mix-cb", CircuitState.CLOSED, "open")

        metric = pm_module._metrics.get("resilience_circuit_breaker_state_changes_total")
        key = (
            ("breaker_name", "mix-cb"),
            ("from_state", "closed"),
            ("to_state", "open"),
        )
        child = metric._children.get(key)
        assert child is not None
        child.inc.assert_called_once()


# ============================================================================
# retry_attempt Tests
# ============================================================================


class TestRetryAttempt:
    """Tests for retry_attempt metric recording."""

    def test_creates_counter_and_histogram(self):
        """Test that retry_attempt creates both Counter and Histogram."""
        pm_module._prometheus_available = True
        FakeCounter = _make_mock_metric_class("Counter")
        FakeHistogram = _make_mock_metric_class("Histogram")

        import sys

        mock_prom = MagicMock()
        mock_prom.Counter = FakeCounter
        mock_prom.Histogram = FakeHistogram
        with patch.dict(sys.modules, {"prometheus_client": mock_prom}):
            retry_attempt("fetch-data", attempt=2, delay=1.5)

        # Verify counter
        counter = pm_module._metrics.get("resilience_retry_attempts_total")
        assert counter is not None
        counter_key = (("operation_name", "fetch-data"),)
        counter_child = counter._children.get(counter_key)
        assert counter_child is not None
        counter_child.inc.assert_called_once()

        # Verify histogram
        histogram = pm_module._metrics.get("resilience_retry_delay_seconds")
        assert histogram is not None
        hist_key = (("operation_name", "fetch-data"),)
        hist_child = histogram._children.get(hist_key)
        assert hist_child is not None
        hist_child.observe.assert_called_once_with(1.5)

    def test_with_exception(self):
        """Test retry_attempt with an exception parameter (should not affect metrics)."""
        pm_module._prometheus_available = True
        FakeCounter = _make_mock_metric_class("Counter")
        FakeHistogram = _make_mock_metric_class("Histogram")

        import sys

        mock_prom = MagicMock()
        mock_prom.Counter = FakeCounter
        mock_prom.Histogram = FakeHistogram
        with patch.dict(sys.modules, {"prometheus_client": mock_prom}):
            retry_attempt("flaky-op", attempt=3, delay=2.0, exception=TimeoutError("slow"))

        counter = pm_module._metrics.get("resilience_retry_attempts_total")
        counter_key = (("operation_name", "flaky-op"),)
        counter._children[counter_key].inc.assert_called_once()


# ============================================================================
# retry_exhausted Tests
# ============================================================================


class TestRetryExhausted:
    """Tests for retry_exhausted metric recording."""

    def test_creates_counter(self):
        """Test that retry_exhausted creates a Counter and increments it."""
        pm_module._prometheus_available = True
        FakeCounter = _make_mock_metric_class("Counter")

        import sys

        mock_prom = MagicMock()
        mock_prom.Counter = FakeCounter
        with patch.dict(sys.modules, {"prometheus_client": mock_prom}):
            retry_exhausted("failing-op", total_attempts=5)

        counter = pm_module._metrics.get("resilience_retry_exhausted_total")
        assert counter is not None
        key = (("operation_name", "failing-op"),)
        child = counter._children.get(key)
        assert child is not None
        child.inc.assert_called_once()

    def test_with_last_exception(self):
        """Test retry_exhausted with last_exception parameter."""
        pm_module._prometheus_available = True
        FakeCounter = _make_mock_metric_class("Counter")

        import sys

        mock_prom = MagicMock()
        mock_prom.Counter = FakeCounter
        with patch.dict(sys.modules, {"prometheus_client": mock_prom}):
            retry_exhausted("dead-op", total_attempts=3, last_exception=ConnectionError("gone"))

        counter = pm_module._metrics.get("resilience_retry_exhausted_total")
        key = (("operation_name", "dead-op"),)
        counter._children[key].inc.assert_called_once()


# ============================================================================
# timeout_occurred Tests
# ============================================================================


class TestTimeoutOccurred:
    """Tests for timeout_occurred metric recording."""

    def test_creates_counter_and_histogram(self):
        """Test that timeout_occurred creates both Counter and Histogram."""
        pm_module._prometheus_available = True
        FakeCounter = _make_mock_metric_class("Counter")
        FakeHistogram = _make_mock_metric_class("Histogram")

        import sys

        mock_prom = MagicMock()
        mock_prom.Counter = FakeCounter
        mock_prom.Histogram = FakeHistogram
        with patch.dict(sys.modules, {"prometheus_client": mock_prom}):
            timeout_occurred("slow-query", timeout_seconds=30.0)

        # Verify counter
        counter = pm_module._metrics.get("resilience_timeouts_total")
        assert counter is not None
        counter_key = (("operation_name", "slow-query"),)
        counter._children[counter_key].inc.assert_called_once()

        # Verify histogram
        histogram = pm_module._metrics.get("resilience_timeout_seconds")
        assert histogram is not None
        hist_key = (("operation_name", "slow-query"),)
        histogram._children[hist_key].observe.assert_called_once_with(30.0)


# ============================================================================
# health_status_changed Tests
# ============================================================================


class TestHealthStatusChanged:
    """Tests for health_status_changed metric recording."""

    @pytest.fixture(autouse=True)
    def setup_prometheus(self):
        """Enable prometheus with Gauge mock."""
        pm_module._prometheus_available = True
        self.FakeGauge = _make_mock_metric_class("Gauge")
        yield

    def test_healthy_status(self):
        """Test recording healthy status sets gauge to 1."""
        import sys

        mock_prom = MagicMock()
        mock_prom.Gauge = self.FakeGauge
        with patch.dict(sys.modules, {"prometheus_client": mock_prom}):
            health_status_changed("database", healthy=True, consecutive_failures=0)

        gauge = pm_module._metrics.get("resilience_health_status")
        assert gauge is not None
        key = (("component_name", "database"),)
        child = gauge._children.get(key)
        assert child is not None
        child.set.assert_called_once_with(1)

    def test_unhealthy_status(self):
        """Test recording unhealthy status sets gauge to 0."""
        import sys

        mock_prom = MagicMock()
        mock_prom.Gauge = self.FakeGauge
        with patch.dict(sys.modules, {"prometheus_client": mock_prom}):
            health_status_changed("cache", healthy=False, consecutive_failures=5)

        gauge = pm_module._metrics.get("resilience_health_status")
        key = (("component_name", "cache"),)
        gauge._children[key].set.assert_called_once_with(0)

    def test_consecutive_failures_gauge(self):
        """Test that consecutive failures are recorded in a separate gauge."""
        import sys

        mock_prom = MagicMock()
        mock_prom.Gauge = self.FakeGauge
        with patch.dict(sys.modules, {"prometheus_client": mock_prom}):
            health_status_changed("redis", healthy=False, consecutive_failures=7)

        failures_gauge = pm_module._metrics.get("resilience_health_consecutive_failures")
        assert failures_gauge is not None
        key = (("component_name", "redis"),)
        child = failures_gauge._children.get(key)
        assert child is not None
        child.set.assert_called_once_with(7)

    def test_default_consecutive_failures(self):
        """Test default consecutive_failures is 0."""
        import sys

        mock_prom = MagicMock()
        mock_prom.Gauge = self.FakeGauge
        with patch.dict(sys.modules, {"prometheus_client": mock_prom}):
            health_status_changed("api", healthy=True)

        failures_gauge = pm_module._metrics.get("resilience_health_consecutive_failures")
        key = (("component_name", "api"),)
        failures_gauge._children[key].set.assert_called_once_with(0)

    def test_creates_two_gauges(self):
        """Test that health_status_changed creates both health and failures gauges."""
        import sys

        mock_prom = MagicMock()
        mock_prom.Gauge = self.FakeGauge
        with patch.dict(sys.modules, {"prometheus_client": mock_prom}):
            health_status_changed("worker", healthy=True, consecutive_failures=0)

        assert "resilience_health_status" in pm_module._metrics
        assert "resilience_health_consecutive_failures" in pm_module._metrics


# ============================================================================
# operation_duration Tests
# ============================================================================


class TestOperationDuration:
    """Tests for operation_duration metric recording."""

    def test_records_successful_operation(self):
        """Test recording a successful operation duration."""
        pm_module._prometheus_available = True
        FakeHistogram = _make_mock_metric_class("Histogram")

        import sys

        mock_prom = MagicMock()
        mock_prom.Histogram = FakeHistogram
        with patch.dict(sys.modules, {"prometheus_client": mock_prom}):
            operation_duration("query-db", duration_seconds=0.45, success=True)

        histogram = pm_module._metrics.get("resilience_operation_duration_seconds")
        assert histogram is not None
        key = (("operation_name", "query-db"), ("success", "true"))
        child = histogram._children.get(key)
        assert child is not None
        child.observe.assert_called_once_with(0.45)

    def test_records_failed_operation(self):
        """Test recording a failed operation duration."""
        pm_module._prometheus_available = True
        FakeHistogram = _make_mock_metric_class("Histogram")

        import sys

        mock_prom = MagicMock()
        mock_prom.Histogram = FakeHistogram
        with patch.dict(sys.modules, {"prometheus_client": mock_prom}):
            operation_duration("process-request", duration_seconds=5.2, success=False)

        histogram = pm_module._metrics.get("resilience_operation_duration_seconds")
        key = (("operation_name", "process-request"), ("success", "false"))
        child = histogram._children.get(key)
        assert child is not None
        child.observe.assert_called_once_with(5.2)

    def test_default_success_is_true(self):
        """Test that success defaults to True."""
        pm_module._prometheus_available = True
        FakeHistogram = _make_mock_metric_class("Histogram")

        import sys

        mock_prom = MagicMock()
        mock_prom.Histogram = FakeHistogram
        with patch.dict(sys.modules, {"prometheus_client": mock_prom}):
            operation_duration("default-op", duration_seconds=1.0)

        histogram = pm_module._metrics.get("resilience_operation_duration_seconds")
        key = (("operation_name", "default-op"), ("success", "true"))
        child = histogram._children.get(key)
        assert child is not None
        child.observe.assert_called_once_with(1.0)


# ============================================================================
# create_metrics_callbacks Tests
# ============================================================================


class TestCreateMetricsCallbacks:
    """Tests for create_metrics_callbacks factory function."""

    def test_returns_dict(self):
        """Test that create_metrics_callbacks returns a dictionary."""
        result = create_metrics_callbacks()
        assert isinstance(result, dict)

    def test_contains_all_expected_keys(self):
        """Test that the callback dict contains all expected keys."""
        result = create_metrics_callbacks()
        expected_keys = {
            "on_circuit_state_change",
            "on_retry",
            "on_retry_exhausted",
            "on_timeout",
            "on_health_change",
            "on_operation_complete",
        }
        assert set(result.keys()) == expected_keys

    def test_values_are_callables(self):
        """Test that all values in the dict are callable."""
        result = create_metrics_callbacks()
        for key, value in result.items():
            assert callable(value), f"Value for '{key}' is not callable"

    def test_maps_to_correct_functions(self):
        """Test that callbacks map to the correct functions."""
        result = create_metrics_callbacks()
        assert result["on_circuit_state_change"] is circuit_breaker_state_changed
        assert result["on_retry"] is retry_attempt
        assert result["on_retry_exhausted"] is retry_exhausted
        assert result["on_timeout"] is timeout_occurred
        assert result["on_health_change"] is health_status_changed
        assert result["on_operation_complete"] is operation_duration


# ============================================================================
# reset_metrics Tests
# ============================================================================


class TestResetMetrics:
    """Tests for reset_metrics clearing stored metrics."""

    def test_clears_empty_metrics(self):
        """Test reset_metrics on already-empty dict is fine."""
        reset_metrics()
        assert len(pm_module._metrics) == 0

    def test_clears_populated_metrics(self):
        """Test reset_metrics clears all stored metrics."""
        pm_module._metrics["metric_a"] = MagicMock()
        pm_module._metrics["metric_b"] = MagicMock()
        assert len(pm_module._metrics) == 2

        reset_metrics()
        assert len(pm_module._metrics) == 0

    def test_metrics_dict_is_reusable_after_reset(self):
        """Test that _metrics dict can be populated again after reset."""
        pm_module._prometheus_available = True
        FakeCounter = _make_mock_metric_class("Counter")

        pm_module._metrics["old_metric"] = MagicMock()
        reset_metrics()

        # Manually add a metric to verify the dict still works
        metric = _get_or_create_metric("new_metric", FakeCounter, "desc", ["l"])
        assert metric is not None
        assert "new_metric" in pm_module._metrics
        assert "old_metric" not in pm_module._metrics


# ============================================================================
# Integration-style Tests
# ============================================================================


class TestMultipleMetricOperations:
    """Tests verifying multiple metric operations interact correctly."""

    def test_multiple_functions_share_metrics_dict(self):
        """Test that different metric functions all store in _metrics."""
        pm_module._prometheus_available = True
        FakeCounter = _make_mock_metric_class("Counter")
        FakeHistogram = _make_mock_metric_class("Histogram")
        FakeGauge = _make_mock_metric_class("Gauge")

        import sys

        mock_prom = MagicMock()
        mock_prom.Counter = FakeCounter
        mock_prom.Histogram = FakeHistogram
        mock_prom.Gauge = FakeGauge

        with patch.dict(sys.modules, {"prometheus_client": mock_prom}):
            circuit_breaker_state_changed("cb1", "closed", "open")
            retry_attempt("op1", attempt=1, delay=0.5)
            retry_exhausted("op2", total_attempts=3)
            timeout_occurred("op3", timeout_seconds=10.0)
            health_status_changed("comp1", healthy=True)
            operation_duration("op4", duration_seconds=2.0)

        # All metrics should be in _metrics
        expected_metric_names = {
            "resilience_circuit_breaker_state_changes_total",
            "resilience_retry_attempts_total",
            "resilience_retry_delay_seconds",
            "resilience_retry_exhausted_total",
            "resilience_timeouts_total",
            "resilience_timeout_seconds",
            "resilience_health_status",
            "resilience_health_consecutive_failures",
            "resilience_operation_duration_seconds",
        }
        assert set(pm_module._metrics.keys()) == expected_metric_names

    def test_reset_then_recreate(self):
        """Test that metrics can be recreated after reset."""
        pm_module._prometheus_available = True
        FakeCounter = _make_mock_metric_class("Counter")

        import sys

        mock_prom = MagicMock()
        mock_prom.Counter = FakeCounter

        with patch.dict(sys.modules, {"prometheus_client": mock_prom}):
            retry_exhausted("op1", total_attempts=3)
            assert "resilience_retry_exhausted_total" in pm_module._metrics
            old_metric = pm_module._metrics["resilience_retry_exhausted_total"]

            reset_metrics()
            assert len(pm_module._metrics) == 0

            retry_exhausted("op1", total_attempts=5)
            assert "resilience_retry_exhausted_total" in pm_module._metrics
            new_metric = pm_module._metrics["resilience_retry_exhausted_total"]
            # After reset, a fresh metric instance should be created
            assert new_metric is not old_metric
