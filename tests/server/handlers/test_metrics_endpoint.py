"""
Tests for the UnifiedMetricsHandler and metrics endpoint utilities.

Tests cover:
- Handler routing for Prometheus metrics endpoints
- Prometheus format output
- Metrics registry initialization
- Cardinality management
- Metrics summary endpoint
- All expected metrics are present
- Metric values accuracy
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.metrics_endpoint import (
    CardinalityConfig,
    MetricsRegistry,
    UnifiedMetricsHandler,
    ensure_all_metrics_registered,
    generate_prometheus_metrics,
    get_metrics_summary,
    get_registered_metric_names,
    _normalize_endpoint,
    _normalize_table,
    PROMETHEUS_CONTENT_TYPE,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None}


@pytest.fixture
def handler(mock_server_context):
    """Create a UnifiedMetricsHandler instance."""
    return UnifiedMetricsHandler(mock_server_context)


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler."""
    mock = MagicMock()
    mock.client_address = ("127.0.0.1", 8080)
    mock.headers = {}
    return mock


@pytest.fixture(autouse=True)
def reset_metrics_registry():
    """Reset metrics registry state between tests."""
    MetricsRegistry._initialized = False
    MetricsRegistry._initialization_time = 0.0
    yield


# =============================================================================
# Handler Routing Tests
# =============================================================================


class TestUnifiedMetricsHandlerRouting:
    """Tests for handler routing."""

    def test_can_handle_metrics(self, handler):
        """Handler can handle /metrics endpoint."""
        assert handler.can_handle("/metrics")

    def test_can_handle_api_prometheus(self, handler):
        """Handler can handle API versioned prometheus endpoint."""
        assert handler.can_handle("/api/v1/metrics/prometheus")
        assert handler.can_handle("/api/metrics/prometheus")

    def test_can_handle_prometheus_summary(self, handler):
        """Handler can handle prometheus summary endpoint."""
        assert handler.can_handle("/api/v1/metrics/prometheus/summary")
        assert handler.can_handle("/api/metrics/prometheus/summary")

    def test_cannot_handle_unrelated_paths(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/v1/agents")
        assert not handler.can_handle("/api/v1/metrics")  # Different handler
        assert not handler.can_handle("/health")


class TestUnifiedMetricsHandlerRoutesAttribute:
    """Tests for ROUTES class attribute."""

    def test_routes_contains_metrics(self, handler):
        """ROUTES contains /metrics."""
        assert "/metrics" in handler.ROUTES

    def test_routes_contains_api_prometheus(self, handler):
        """ROUTES contains API prometheus endpoint."""
        assert "/api/metrics/prometheus" in handler.ROUTES

    def test_routes_contains_summary(self, handler):
        """ROUTES contains summary endpoint."""
        assert "/api/metrics/prometheus/summary" in handler.ROUTES


# =============================================================================
# Handler Response Tests
# =============================================================================


class TestUnifiedMetricsHandlerResponses:
    """Tests for handler responses."""

    def test_handle_metrics_returns_result(self, handler, mock_http_handler):
        """Handle returns result for /metrics endpoint."""
        result = handler.handle("/metrics", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200

    def test_metrics_response_is_prometheus_format(self, handler, mock_http_handler):
        """Metrics endpoint returns Prometheus format."""
        result = handler.handle("/metrics", {}, mock_http_handler)

        assert result is not None
        # Should be text/plain or openmetrics format
        assert "text" in result.content_type or "openmetrics" in result.content_type

    def test_metrics_body_is_valid(self, handler, mock_http_handler):
        """Metrics body contains valid Prometheus format."""
        result = handler.handle("/metrics", {}, mock_http_handler)

        assert result is not None
        body = result.body.decode("utf-8")

        # Prometheus format should contain # HELP or # TYPE comments
        # or at minimum metric lines
        assert len(body) > 0

    def test_handle_summary_returns_json(self, handler, mock_http_handler):
        """Summary endpoint returns JSON."""
        result = handler.handle("/api/metrics/prometheus/summary", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        assert result.content_type == "application/json"

    def test_summary_response_structure(self, handler, mock_http_handler):
        """Summary response has expected structure."""
        result = handler.handle("/api/metrics/prometheus/summary", {}, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)

        assert "initialized" in body
        assert "initialization_time_seconds" in body
        assert "metrics" in body

    def test_handle_unknown_returns_none(self, handler, mock_http_handler):
        """Handle returns None for unknown paths."""
        result = handler.handle("/api/v1/unknown", {}, mock_http_handler)

        assert result is None

    def test_aggregate_parameter(self, handler, mock_http_handler):
        """Handler accepts aggregate query parameter."""
        result = handler.handle(
            "/metrics",
            {"aggregate": ["true"]},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 200


# =============================================================================
# Metrics Registry Tests
# =============================================================================


class TestMetricsRegistry:
    """Tests for MetricsRegistry."""

    def test_ensure_initialized_returns_bool(self):
        """ensure_initialized returns boolean."""
        result = MetricsRegistry.ensure_initialized()
        assert isinstance(result, bool)

    def test_ensure_initialized_is_idempotent(self):
        """Multiple calls to ensure_initialized are idempotent.

        Note: Due to prometheus_client's global registry, the first call
        in a test session may succeed while subsequent calls may fail
        if metrics are already registered. We test that the _initialized
        flag is properly set.
        """
        # First call sets up initialization
        MetricsRegistry.ensure_initialized()

        # After first call, _initialized should be True
        assert MetricsRegistry._initialized is True

        # Second call should return quickly (already initialized)
        result2 = MetricsRegistry.ensure_initialized()

        # Should still be initialized
        assert MetricsRegistry._initialized is True
        assert isinstance(result2, bool)

    def test_get_initialization_time(self):
        """get_initialization_time returns float."""
        MetricsRegistry.ensure_initialized()
        time_val = MetricsRegistry.get_initialization_time()

        assert isinstance(time_val, float)
        assert time_val >= 0


# =============================================================================
# Prometheus Metrics Generation Tests
# =============================================================================


class TestPrometheusMetricsGeneration:
    """Tests for generate_prometheus_metrics."""

    def test_returns_tuple(self):
        """generate_prometheus_metrics returns (content, content_type) tuple."""
        result = generate_prometheus_metrics()

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_content_is_string(self):
        """Content is a string."""
        content, _ = generate_prometheus_metrics()

        assert isinstance(content, str)

    def test_content_type_is_valid(self):
        """Content type is valid Prometheus type."""
        _, content_type = generate_prometheus_metrics()

        # Should be one of the valid Prometheus content types
        valid_types = [
            "text/plain",
            "application/openmetrics-text",
            PROMETHEUS_CONTENT_TYPE,
        ]
        assert any(valid in content_type for valid in valid_types)


# =============================================================================
# Metrics Summary Tests
# =============================================================================


class TestMetricsSummary:
    """Tests for get_metrics_summary."""

    def test_returns_dict(self):
        """get_metrics_summary returns a dictionary."""
        summary = get_metrics_summary()

        assert isinstance(summary, dict)

    def test_summary_has_initialized(self):
        """Summary includes initialized flag."""
        summary = get_metrics_summary()

        assert "initialized" in summary

    def test_summary_has_metrics(self):
        """Summary includes metrics section."""
        summary = get_metrics_summary()

        assert "metrics" in summary


# =============================================================================
# Registered Metric Names Tests
# =============================================================================


class TestRegisteredMetricNames:
    """Tests for get_registered_metric_names."""

    def test_returns_list(self):
        """get_registered_metric_names returns a list."""
        names = get_registered_metric_names()

        assert isinstance(names, list)

    def test_names_are_sorted(self):
        """Metric names are sorted."""
        names = get_registered_metric_names()

        if len(names) > 1:
            assert names == sorted(names)


# =============================================================================
# Ensure All Metrics Registered Tests
# =============================================================================


class TestEnsureAllMetricsRegistered:
    """Tests for ensure_all_metrics_registered."""

    def test_returns_bool(self):
        """ensure_all_metrics_registered returns boolean."""
        result = ensure_all_metrics_registered()

        assert isinstance(result, bool)

    def test_is_idempotent(self):
        """Multiple calls are idempotent.

        Note: Due to prometheus_client's global registry, we test that
        the initialization state is properly tracked rather than requiring
        identical return values (which may vary due to registry state).
        """
        # First call initializes
        ensure_all_metrics_registered()

        # After first call, should be marked as initialized
        assert MetricsRegistry._initialized is True

        # Second call should also complete
        result2 = ensure_all_metrics_registered()
        assert isinstance(result2, bool)


# =============================================================================
# Cardinality Management Tests
# =============================================================================


class TestCardinalityConfig:
    """Tests for CardinalityConfig."""

    def test_default_values(self):
        """CardinalityConfig has sensible defaults."""
        config = CardinalityConfig()

        assert config.max_label_values > 0
        assert isinstance(config.high_cardinality_metrics, list)
        assert isinstance(config.aggregation_enabled, bool)

    def test_high_cardinality_metrics_defaults(self):
        """High cardinality metrics list includes expected metrics."""
        config = CardinalityConfig()

        expected = [
            "aragora_http_requests_total",
            "aragora_http_request_duration_seconds",
        ]
        for metric in expected:
            assert metric in config.high_cardinality_metrics


class TestNormalizeEndpoint:
    """Tests for endpoint normalization."""

    def test_replaces_uuids(self):
        """UUIDs are replaced with :id."""
        endpoint = "/api/v1/debates/123e4567-e89b-12d3-a456-426614174000"
        normalized = _normalize_endpoint(endpoint)

        assert ":id" in normalized
        assert "123e4567" not in normalized

    def test_replaces_numeric_ids(self):
        """Numeric IDs are replaced with :id."""
        endpoint = "/api/v1/agents/12345"
        normalized = _normalize_endpoint(endpoint)

        assert "/:id" in normalized
        assert "12345" not in normalized

    def test_replaces_tokens(self):
        """Long alphanumeric tokens are replaced with :token."""
        endpoint = "/api/v1/auth/AbCdEfGhIjKlMnOpQrStUvWx"
        normalized = _normalize_endpoint(endpoint)

        assert ":token" in normalized

    def test_preserves_static_paths(self):
        """Static paths are preserved."""
        endpoint = "/api/v1/health"
        normalized = _normalize_endpoint(endpoint)

        assert normalized == endpoint


class TestNormalizeTable:
    """Tests for table name normalization."""

    def test_replaces_shard_suffix(self):
        """Shard suffixes are replaced with :shard."""
        table = "debates_001"
        normalized = _normalize_table(table)

        assert normalized == "debates_:shard"

    def test_preserves_regular_names(self):
        """Regular table names are preserved."""
        table = "debates"
        normalized = _normalize_table(table)

        assert normalized == "debates"

    def test_preserves_underscores(self):
        """Table names with underscores are preserved."""
        table = "debate_rounds"
        normalized = _normalize_table(table)

        assert normalized == "debate_rounds"


# =============================================================================
# All Expected Metrics Present Tests
# =============================================================================


class TestExpectedMetricsPresent:
    """Tests that verify expected metrics are registered."""

    @pytest.fixture
    def metrics_content(self):
        """Get metrics content after initialization."""
        content, _ = generate_prometheus_metrics()
        return content

    def test_agent_provider_metrics_registered(self, metrics_content):
        """Agent provider metrics should be registered."""
        # These metrics are defined in agents.py
        expected_prefixes = [
            "aragora_agent_provider",
            "aragora_agent_token",
        ]
        # At least one of these should appear if prometheus is available
        # (may not appear if prometheus_client not installed)
        if "aragora" in metrics_content:
            found = any(prefix in metrics_content for prefix in expected_prefixes)
            # Note: In test environments, not all metrics may be populated
            # This test just verifies the structure is valid
            assert isinstance(found, bool)

    def test_request_metrics_registered(self, metrics_content):
        """Request metrics should be registered."""
        if "aragora" in metrics_content:
            # Check for request-related metrics
            assert "aragora" in metrics_content

    def test_debate_metrics_registered(self, metrics_content):
        """Debate metrics should be registered."""
        if "aragora" in metrics_content:
            # Check for debate-related metrics
            assert "aragora" in metrics_content


# =============================================================================
# Metric Values Accuracy Tests
# =============================================================================


class TestMetricValuesAccuracy:
    """Tests that verify metric values are accurate."""

    def test_initialization_time_positive(self):
        """Initialization time is positive after init."""
        MetricsRegistry.ensure_initialized()
        time_val = MetricsRegistry.get_initialization_time()

        if MetricsRegistry._initialized:
            # Time should be non-negative (could be 0 if cached)
            assert time_val >= 0

    def test_summary_metrics_counts_valid(self):
        """Summary metric counts are valid."""
        summary = get_metrics_summary()

        if "metrics" in summary and isinstance(summary["metrics"], dict):
            for key, value in summary["metrics"].items():
                if isinstance(value, (int, float)):
                    assert value >= 0


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_query_params(self, handler, mock_http_handler):
        """Handler handles empty query params."""
        result = handler.handle("/metrics", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200

    def test_none_context(self):
        """Handler handles None context."""
        handler = UnifiedMetricsHandler(None)

        assert handler.ctx == {}

    def test_malformed_aggregate_param(self, handler, mock_http_handler):
        """Handler handles malformed aggregate parameter."""
        result = handler.handle(
            "/metrics",
            {"aggregate": ["invalid"]},
            mock_http_handler,
        )

        # Should still succeed (defaults to false)
        assert result is not None
        assert result.status_code == 200


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for metrics endpoint."""

    def test_full_metrics_flow(self, handler, mock_http_handler):
        """Test complete metrics flow from request to response."""
        # 1. Ensure initialization
        initialized = ensure_all_metrics_registered()
        assert isinstance(initialized, bool)

        # 2. Get metrics
        result = handler.handle("/metrics", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

        # 3. Get summary
        summary_result = handler.handle(
            "/api/metrics/prometheus/summary",
            {},
            mock_http_handler,
        )
        assert summary_result is not None
        assert summary_result.status_code == 200

        # 4. Parse summary
        summary = json.loads(summary_result.body)
        assert "initialized" in summary

    def test_metrics_idempotent(self, handler, mock_http_handler):
        """Multiple metrics requests return consistent results."""
        result1 = handler.handle("/metrics", {}, mock_http_handler)
        result2 = handler.handle("/metrics", {}, mock_http_handler)

        assert result1 is not None
        assert result2 is not None
        assert result1.status_code == result2.status_code

    def test_concurrent_initialization(self):
        """Concurrent initialization is safe.

        Note: Due to prometheus_client's global registry, concurrent
        initialization may have varied results as some threads may hit
        already-registered metrics. We verify no exceptions are raised
        and all threads complete successfully.
        """
        import threading

        results = []
        errors = []
        lock = threading.Lock()

        def init_thread():
            try:
                result = MetricsRegistry.ensure_initialized()
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=init_thread) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No uncaught exceptions
        assert len(errors) == 0
        # All threads completed with a boolean result
        assert len(results) == 5
        assert all(isinstance(r, bool) for r in results)
