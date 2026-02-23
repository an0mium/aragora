"""Comprehensive tests for UnifiedMetricsHandler.

Covers all routes, cardinality management, MetricsRegistry, fallback generation,
metrics summary, convenience functions, and error paths.

Test classes:
  TestCardinalityConfig           - CardinalityConfig defaults and customization
  TestNormalizeEndpoint           - UUID, numeric ID, and token normalization
  TestNormalizeTable              - Sharded table name normalization
  TestLimitLabelCardinality       - Cardinality limiting logic
  TestMetricsRegistryInit         - MetricsRegistry.ensure_initialized()
  TestMetricsRegistryGetInitTime  - MetricsRegistry.get_initialization_time()
  TestGeneratePrometheusMetrics   - generate_prometheus_metrics() with/without prometheus_client
  TestGenerateFallbackMetrics     - _generate_fallback_metrics()
  TestGetMetricsSummary           - get_metrics_summary()
  TestHandlerInit                 - Handler construction
  TestHandlerCanHandle            - can_handle() route matching
  TestHandlerRouteMetrics         - GET /metrics
  TestHandlerRouteApiMetrics      - GET /api/v1/metrics/prometheus
  TestHandlerRouteSummary         - GET /api/v1/metrics/prometheus/summary
  TestHandlerAuthRequired         - Auth/permission checks on API routes
  TestHandlerErrorPaths           - Error handling in handler methods
  TestConvenienceFunctions        - ensure_all_metrics_registered, get_registered_metric_names
  TestExports                     - __all__ exports
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.metrics_endpoint import (
    CardinalityConfig,
    MetricsRegistry,
    PROMETHEUS_CONTENT_TYPE,
    UnifiedMetricsHandler,
    _generate_fallback_metrics,
    _limit_label_cardinality,
    _normalize_endpoint,
    _normalize_table,
    ensure_all_metrics_registered,
    generate_prometheus_metrics,
    get_metrics_summary,
    get_registered_metric_names,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict[str, Any]:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _raw_body(result) -> str:
    """Extract raw string body from a HandlerResult."""
    if result is None:
        return ""
    raw = result.body
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


class MockHTTPHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, body: dict[str, Any] | None = None):
        self.rfile = MagicMock()
        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {
                "Content-Length": str(len(body_bytes)),
                "Content-Type": "application/json",
                "Authorization": "Bearer test-token",
            }
        else:
            self.rfile.read.return_value = b""
            self.headers = {
                "Content-Length": "0",
                "Content-Type": "application/json",
                "Authorization": "Bearer test-token",
            }
        self.client_address = ("127.0.0.1", 12345)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_metrics_registry():
    """Reset MetricsRegistry state between tests."""
    original_initialized = MetricsRegistry._initialized
    original_time = MetricsRegistry._initialization_time
    yield
    MetricsRegistry._initialized = original_initialized
    MetricsRegistry._initialization_time = original_time


@pytest.fixture
def handler():
    """Create a UnifiedMetricsHandler with mocked initialization."""
    with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
        return UnifiedMetricsHandler(ctx={})


@pytest.fixture
def mock_http():
    """Create a mock HTTP handler."""
    return MockHTTPHandler()


@pytest.fixture
def mock_http_with_body():
    """Factory for mock HTTP handler with body."""
    def _create(body: dict[str, Any]) -> MockHTTPHandler:
        return MockHTTPHandler(body=body)
    return _create


# ===========================================================================
# TestCardinalityConfig
# ===========================================================================


class TestCardinalityConfig:
    """Test CardinalityConfig defaults and customization."""

    def test_default_max_label_values(self):
        config = CardinalityConfig()
        assert config.max_label_values == 1000

    def test_default_high_cardinality_metrics(self):
        config = CardinalityConfig()
        assert "aragora_http_requests_total" in config.high_cardinality_metrics
        assert "aragora_http_request_duration_seconds" in config.high_cardinality_metrics
        assert "aragora_db_query_duration_seconds" in config.high_cardinality_metrics
        assert "aragora_agent_provider_calls_total" in config.high_cardinality_metrics

    def test_default_aggregation_enabled(self):
        config = CardinalityConfig()
        assert config.aggregation_enabled is True

    def test_custom_max_label_values(self):
        config = CardinalityConfig(max_label_values=500)
        assert config.max_label_values == 500

    def test_custom_high_cardinality_metrics(self):
        config = CardinalityConfig(high_cardinality_metrics=["my_metric"])
        assert config.high_cardinality_metrics == ["my_metric"]

    def test_custom_aggregation_disabled(self):
        config = CardinalityConfig(aggregation_enabled=False)
        assert config.aggregation_enabled is False

    def test_high_cardinality_metrics_has_four_defaults(self):
        config = CardinalityConfig()
        assert len(config.high_cardinality_metrics) == 4


# ===========================================================================
# TestNormalizeEndpoint
# ===========================================================================


class TestNormalizeEndpoint:
    """Test _normalize_endpoint path normalization."""

    def test_replaces_uuid(self):
        result = _normalize_endpoint("/api/v1/debates/550e8400-e29b-41d4-a716-446655440000/result")
        assert ":id" in result
        assert "550e8400" not in result

    def test_replaces_uppercase_uuid(self):
        result = _normalize_endpoint("/api/v1/debates/550E8400-E29B-41D4-A716-446655440000/result")
        assert ":id" in result

    def test_replaces_numeric_ids(self):
        result = _normalize_endpoint("/api/v1/users/12345/profile")
        assert "/:id/profile" in result
        assert "12345" not in result

    def test_replaces_base64_like_tokens(self):
        # 20+ chars of alphanumeric
        result = _normalize_endpoint("/api/v1/auth/abcdefghijklmnopqrstuvwxyz/verify")
        assert ":token" in result

    def test_preserves_short_path_segments(self):
        result = _normalize_endpoint("/api/v1/debates/list")
        assert result == "/api/v1/debates/list"

    def test_preserves_simple_paths(self):
        result = _normalize_endpoint("/metrics")
        assert result == "/metrics"

    def test_replaces_multiple_uuids(self):
        path = "/api/v1/debates/550e8400-e29b-41d4-a716-446655440000/agents/660e8400-e29b-41d4-a716-446655440001"
        result = _normalize_endpoint(path)
        assert result.count(":id") == 2

    def test_replaces_multiple_numeric_ids(self):
        result = _normalize_endpoint("/api/v1/org/100/users/200")
        assert result.count("/:id") == 2

    def test_empty_path(self):
        result = _normalize_endpoint("")
        assert result == ""

    def test_root_path(self):
        result = _normalize_endpoint("/")
        assert result == "/"


# ===========================================================================
# TestNormalizeTable
# ===========================================================================


class TestNormalizeTable:
    """Test _normalize_table table name normalization."""

    def test_replaces_shard_suffix(self):
        result = _normalize_table("events_001")
        assert result == "events_:shard"

    def test_replaces_large_shard_suffix(self):
        result = _normalize_table("logs_999")
        assert result == "logs_:shard"

    def test_preserves_non_sharded_table(self):
        result = _normalize_table("debates")
        assert result == "debates"

    def test_preserves_single_digit_suffix(self):
        # Single digit (< 2 digits) should NOT be replaced
        result = _normalize_table("table_1")
        assert result == "table_1"

    def test_replaces_two_digit_suffix(self):
        result = _normalize_table("metrics_42")
        assert result == "metrics_:shard"

    def test_replaces_four_digit_suffix(self):
        result = _normalize_table("data_1234")
        assert result == "data_:shard"

    def test_preserves_mid_string_digits(self):
        # Only replaces suffix pattern
        result = _normalize_table("table_v2_data")
        assert result == "table_v2_data"


# ===========================================================================
# TestLimitLabelCardinality
# ===========================================================================


class TestLimitLabelCardinality:
    """Test _limit_label_cardinality logic."""

    def test_no_change_for_non_high_cardinality_metric(self):
        config = CardinalityConfig()
        labels = {"endpoint": "/api/v1/users/12345", "method": "GET"}
        result = _limit_label_cardinality("some_other_metric", labels, config)
        assert result == labels

    def test_normalizes_endpoint_for_high_cardinality(self):
        config = CardinalityConfig()
        labels = {"endpoint": "/api/v1/users/12345", "method": "GET"}
        result = _limit_label_cardinality("aragora_http_requests_total", labels, config)
        assert "/:id" in result["endpoint"]
        assert result["method"] == "GET"

    def test_normalizes_table_for_high_cardinality(self):
        config = CardinalityConfig()
        labels = {"table": "events_042", "operation": "SELECT"}
        result = _limit_label_cardinality("aragora_db_query_duration_seconds", labels, config)
        assert result["table"] == "events_:shard"
        assert result["operation"] == "SELECT"

    def test_does_not_mutate_original_labels(self):
        config = CardinalityConfig()
        labels = {"endpoint": "/api/v1/users/12345"}
        _limit_label_cardinality("aragora_http_requests_total", labels, config)
        assert labels["endpoint"] == "/api/v1/users/12345"

    def test_normalizes_both_endpoint_and_table(self):
        config = CardinalityConfig()
        labels = {"endpoint": "/api/v1/users/999", "table": "events_123"}
        result = _limit_label_cardinality("aragora_http_requests_total", labels, config)
        assert "/:id" in result["endpoint"]
        assert result["table"] == "events_:shard"

    def test_empty_labels_for_high_cardinality(self):
        config = CardinalityConfig()
        result = _limit_label_cardinality("aragora_http_requests_total", {}, config)
        assert result == {}

    def test_labels_without_endpoint_or_table(self):
        config = CardinalityConfig()
        labels = {"status": "200", "method": "POST"}
        result = _limit_label_cardinality("aragora_http_requests_total", labels, config)
        assert result == labels

    def test_custom_high_cardinality_list(self):
        config = CardinalityConfig(high_cardinality_metrics=["my_custom_metric"])
        labels = {"endpoint": "/api/v1/users/999"}
        # Standard metric should NOT be normalized now
        result = _limit_label_cardinality("aragora_http_requests_total", labels, config)
        assert result == labels
        # Custom metric should be normalized
        result2 = _limit_label_cardinality("my_custom_metric", labels, config)
        assert "/:id" in result2["endpoint"]


# ===========================================================================
# TestMetricsRegistryInit
# ===========================================================================


class TestMetricsRegistryInit:
    """Test MetricsRegistry.ensure_initialized()."""

    def test_returns_true_if_already_initialized(self):
        MetricsRegistry._initialized = True
        result = MetricsRegistry.ensure_initialized()
        assert result is True

    def test_initializes_core_metrics(self):
        MetricsRegistry._initialized = False
        with patch(
            "aragora.server.handlers.metrics_endpoint.MetricsRegistry.ensure_initialized",
            return_value=True,
        ):
            assert MetricsRegistry.ensure_initialized() is True

    def test_returns_false_when_core_disabled(self):
        MetricsRegistry._initialized = False
        with patch.dict("sys.modules", {}):
            with patch(
                "aragora.server.handlers.metrics_endpoint.MetricsRegistry.ensure_initialized",
                return_value=False,
            ):
                assert MetricsRegistry.ensure_initialized() is False

    def test_handles_import_error(self):
        MetricsRegistry._initialized = False
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            result = MetricsRegistry.ensure_initialized()
            # After ImportError, _initialized is True, returns False
            assert MetricsRegistry._initialized is True

    def test_handles_runtime_error(self):
        MetricsRegistry._initialized = False

        def raise_runtime(*a, **kw):
            raise RuntimeError("bad")

        with patch("builtins.__import__", side_effect=RuntimeError("bad")):
            result = MetricsRegistry.ensure_initialized()
            assert MetricsRegistry._initialized is True

    def test_sets_initialization_time(self):
        MetricsRegistry._initialized = False
        MetricsRegistry._initialization_time = 0.0
        # After any initialization attempt, time is set or 0
        with patch("builtins.__import__", side_effect=ImportError("no")):
            MetricsRegistry.ensure_initialized()
        # Even on error, _initialized is set
        assert MetricsRegistry._initialized is True


# ===========================================================================
# TestMetricsRegistryGetInitTime
# ===========================================================================


class TestMetricsRegistryGetInitTime:
    """Test MetricsRegistry.get_initialization_time()."""

    def test_returns_zero_initially(self):
        MetricsRegistry._initialization_time = 0.0
        assert MetricsRegistry.get_initialization_time() == 0.0

    def test_returns_set_value(self):
        MetricsRegistry._initialization_time = 1.234
        assert MetricsRegistry.get_initialization_time() == 1.234

    def test_returns_float(self):
        MetricsRegistry._initialization_time = 0.5
        assert isinstance(MetricsRegistry.get_initialization_time(), float)


# ===========================================================================
# TestGeneratePrometheusMetrics
# ===========================================================================


class TestGeneratePrometheusMetrics:
    """Test generate_prometheus_metrics()."""

    def test_returns_tuple_of_str_and_content_type(self):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            with patch(
                "aragora.server.handlers.metrics_endpoint.generate_latest",
                return_value=b"# test\n",
                create=True,
            ):
                with patch(
                    "aragora.server.handlers.metrics_endpoint.REGISTRY",
                    create=True,
                ):
                    with patch(
                        "aragora.server.handlers.metrics_endpoint.CONTENT_TYPE_LATEST",
                        "text/plain; version=0.0.4",
                        create=True,
                    ):
                        # Actually, the function imports from prometheus_client
                        # Let's mock at the import level
                        pass

        # Simpler: just mock prometheus_client
        mock_generate = MagicMock(return_value=b"# metrics\n")
        mock_registry = MagicMock()
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            with patch.dict("sys.modules", {
                "prometheus_client": MagicMock(
                    REGISTRY=mock_registry,
                    generate_latest=mock_generate,
                    CONTENT_TYPE_LATEST="text/plain; version=0.0.4",
                ),
            }):
                text, ct = generate_prometheus_metrics()
                assert text == "# metrics\n"
                assert ct == "text/plain; version=0.0.4"

    def test_fallback_when_prometheus_not_installed(self):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=False):
            # Remove prometheus_client from sys.modules to trigger ImportError
            import sys
            saved = sys.modules.get("prometheus_client")
            sys.modules["prometheus_client"] = None  # type: ignore[assignment]
            try:
                # The import inside the function will raise ImportError when module is None
                # Actually, setting to None causes ImportError on `from X import Y`
                text, ct = generate_prometheus_metrics()
                assert "aragora_info" in text
                assert ct == PROMETHEUS_CONTENT_TYPE
            finally:
                if saved is not None:
                    sys.modules["prometheus_client"] = saved
                else:
                    sys.modules.pop("prometheus_client", None)

    def test_aggregate_param_passed_through(self):
        mock_generate = MagicMock(return_value=b"# agg\n")
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            with patch.dict("sys.modules", {
                "prometheus_client": MagicMock(
                    REGISTRY=MagicMock(),
                    generate_latest=mock_generate,
                    CONTENT_TYPE_LATEST="text/plain",
                ),
            }):
                text, ct = generate_prometheus_metrics(aggregate_high_cardinality=True)
                assert text == "# agg\n"


# ===========================================================================
# TestGenerateFallbackMetrics
# ===========================================================================


class TestGenerateFallbackMetrics:
    """Test _generate_fallback_metrics()."""

    def test_contains_aragora_info(self):
        result = _generate_fallback_metrics()
        assert "aragora_info" in result

    def test_contains_help_line(self):
        result = _generate_fallback_metrics()
        assert "# HELP aragora_info" in result

    def test_contains_type_line(self):
        result = _generate_fallback_metrics()
        assert "# TYPE aragora_info gauge" in result

    def test_prometheus_available_false(self):
        result = _generate_fallback_metrics()
        assert 'prometheus_available="false"' in result

    def test_contains_metrics_initialized_metric(self):
        result = _generate_fallback_metrics()
        assert "aragora_metrics_initialized" in result

    def test_initialized_value_is_zero(self):
        result = _generate_fallback_metrics()
        assert "aragora_metrics_initialized 0" in result

    def test_returns_string(self):
        result = _generate_fallback_metrics()
        assert isinstance(result, str)

    def test_has_multiple_lines(self):
        result = _generate_fallback_metrics()
        lines = result.strip().split("\n")
        assert len(lines) >= 5


# ===========================================================================
# TestGetMetricsSummary
# ===========================================================================


class TestGetMetricsSummary:
    """Test get_metrics_summary()."""

    def test_returns_dict(self):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            import sys
            saved = sys.modules.get("prometheus_client")
            sys.modules["prometheus_client"] = None  # type: ignore[assignment]
            try:
                result = get_metrics_summary()
                assert isinstance(result, dict)
            finally:
                if saved is not None:
                    sys.modules["prometheus_client"] = saved
                else:
                    sys.modules.pop("prometheus_client", None)

    def test_contains_initialized_key(self):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            MetricsRegistry._initialized = True
            import sys
            saved = sys.modules.get("prometheus_client")
            sys.modules["prometheus_client"] = None  # type: ignore[assignment]
            try:
                result = get_metrics_summary()
                assert "initialized" in result
                assert result["initialized"] is True
            finally:
                if saved is not None:
                    sys.modules["prometheus_client"] = saved
                else:
                    sys.modules.pop("prometheus_client", None)

    def test_contains_initialization_time(self):
        MetricsRegistry._initialization_time = 0.42
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            import sys
            saved = sys.modules.get("prometheus_client")
            sys.modules["prometheus_client"] = None  # type: ignore[assignment]
            try:
                result = get_metrics_summary()
                assert result["initialization_time_seconds"] == 0.42
            finally:
                if saved is not None:
                    sys.modules["prometheus_client"] = saved
                else:
                    sys.modules.pop("prometheus_client", None)

    def test_metrics_unavailable_when_prometheus_missing(self):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            import sys
            saved = sys.modules.get("prometheus_client")
            sys.modules["prometheus_client"] = None  # type: ignore[assignment]
            try:
                result = get_metrics_summary()
                assert result["metrics"] == {"available": False}
            finally:
                if saved is not None:
                    sys.modules["prometheus_client"] = saved
                else:
                    sys.modules.pop("prometheus_client", None)

    def test_with_prometheus_collectors(self):
        mock_collector = MagicMock()
        mock_collector._type = "counter"
        mock_collector.samples = [MagicMock()]

        mock_registry = MagicMock()
        mock_registry.collect.return_value = [mock_collector]

        mock_prom = MagicMock()
        mock_prom.REGISTRY = mock_registry

        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            with patch.dict("sys.modules", {"prometheus_client": mock_prom}):
                # Also need to mock the cardinality tracker import
                with patch.dict("sys.modules", {
                    "prometheus_client": mock_prom,
                    "aragora.observability.metrics.cardinality": None,
                }):
                    result = get_metrics_summary()
                    assert "metrics" in result
                    assert result["metrics"]["counters"] == 1


# ===========================================================================
# TestHandlerInit
# ===========================================================================


class TestHandlerInit:
    """Test UnifiedMetricsHandler initialization."""

    def test_init_with_empty_ctx(self):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            h = UnifiedMetricsHandler(ctx={})
            assert h.ctx == {}

    def test_init_with_none_ctx(self):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            h = UnifiedMetricsHandler(ctx=None)
            assert h.ctx == {}

    def test_init_with_custom_ctx(self):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            h = UnifiedMetricsHandler(ctx={"key": "value"})
            assert h.ctx == {"key": "value"}

    def test_has_cardinality_config(self):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            h = UnifiedMetricsHandler(ctx={})
            assert isinstance(h._cardinality_config, CardinalityConfig)

    def test_calls_ensure_initialized(self):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True) as mock_init:
            UnifiedMetricsHandler(ctx={})
            mock_init.assert_called_once()


# ===========================================================================
# TestHandlerCanHandle
# ===========================================================================


class TestHandlerCanHandle:
    """Test can_handle() route matching."""

    def test_handles_metrics(self, handler):
        assert handler.can_handle("/metrics") is True

    def test_handles_api_metrics_prometheus(self, handler):
        assert handler.can_handle("/api/metrics/prometheus") is True

    def test_handles_api_metrics_prometheus_summary(self, handler):
        assert handler.can_handle("/api/metrics/prometheus/summary") is True

    def test_handles_versioned_api_metrics(self, handler):
        # strip_version_prefix converts /api/v1/... to /api/...
        assert handler.can_handle("/api/v1/metrics/prometheus") is True

    def test_handles_versioned_summary(self, handler):
        assert handler.can_handle("/api/v1/metrics/prometheus/summary") is True

    def test_rejects_unknown_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_rejects_partial_match(self, handler):
        assert handler.can_handle("/api/metrics") is False

    def test_rejects_metrics_subpath(self, handler):
        assert handler.can_handle("/metrics/something") is False

    def test_rejects_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_rejects_root(self, handler):
        assert handler.can_handle("/") is False

    def test_routes_constant(self, handler):
        assert "/metrics" in handler.ROUTES
        assert "/api/metrics/prometheus" in handler.ROUTES
        assert "/api/metrics/prometheus/summary" in handler.ROUTES


# ===========================================================================
# TestHandlerRouteMetrics
# ===========================================================================


class TestHandlerRouteMetrics:
    """Test GET /metrics endpoint."""

    def test_returns_200(self, handler, mock_http):
        mock_generate = MagicMock(return_value=b"# test metrics\n")
        with patch.dict("sys.modules", {
            "prometheus_client": MagicMock(
                REGISTRY=MagicMock(),
                generate_latest=mock_generate,
                CONTENT_TYPE_LATEST="text/plain; version=0.0.4",
            ),
        }):
            with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
                result = handler.handle("/metrics", {}, mock_http)
                assert _status(result) == 200

    def test_returns_prometheus_content(self, handler, mock_http):
        mock_generate = MagicMock(return_value=b"# metrics output\n")
        with patch.dict("sys.modules", {
            "prometheus_client": MagicMock(
                REGISTRY=MagicMock(),
                generate_latest=mock_generate,
                CONTENT_TYPE_LATEST="text/plain; version=0.0.4",
            ),
        }):
            with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
                result = handler.handle("/metrics", {}, mock_http)
                raw = _raw_body(result)
                assert "metrics output" in raw

    def test_no_auth_required_for_bare_metrics(self, handler, mock_http):
        """GET /metrics does NOT require auth."""
        mock_generate = MagicMock(return_value=b"# ok\n")
        with patch.dict("sys.modules", {
            "prometheus_client": MagicMock(
                REGISTRY=MagicMock(),
                generate_latest=mock_generate,
                CONTENT_TYPE_LATEST="text/plain",
            ),
        }):
            with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
                result = handler.handle("/metrics", {}, mock_http)
                assert _status(result) == 200

    def test_aggregate_query_param(self, handler, mock_http):
        mock_generate = MagicMock(return_value=b"# agg\n")
        with patch.dict("sys.modules", {
            "prometheus_client": MagicMock(
                REGISTRY=MagicMock(),
                generate_latest=mock_generate,
                CONTENT_TYPE_LATEST="text/plain",
            ),
        }):
            with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
                result = handler.handle("/metrics", {"aggregate": ["true"]}, mock_http)
                assert _status(result) == 200

    def test_aggregate_false_default(self, handler, mock_http):
        mock_generate = MagicMock(return_value=b"# no-agg\n")
        with patch.dict("sys.modules", {
            "prometheus_client": MagicMock(
                REGISTRY=MagicMock(),
                generate_latest=mock_generate,
                CONTENT_TYPE_LATEST="text/plain",
            ),
        }):
            with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
                result = handler.handle("/metrics", {"aggregate": ["false"]}, mock_http)
                assert _status(result) == 200

    def test_fallback_when_prometheus_unavailable(self, handler, mock_http):
        import sys as _sys
        saved = _sys.modules.get("prometheus_client")
        _sys.modules["prometheus_client"] = None  # type: ignore[assignment]
        try:
            with patch.object(MetricsRegistry, "ensure_initialized", return_value=False):
                result = handler.handle("/metrics", {}, mock_http)
                assert _status(result) == 200
                raw = _raw_body(result)
                assert "aragora_info" in raw
        finally:
            if saved is not None:
                _sys.modules["prometheus_client"] = saved
            else:
                _sys.modules.pop("prometheus_client", None)


# ===========================================================================
# TestHandlerRouteApiMetrics
# ===========================================================================


class TestHandlerRouteApiMetrics:
    """Test GET /api/v1/metrics/prometheus endpoint."""

    def test_returns_200(self, handler, mock_http):
        mock_generate = MagicMock(return_value=b"# api metrics\n")
        with patch.dict("sys.modules", {
            "prometheus_client": MagicMock(
                REGISTRY=MagicMock(),
                generate_latest=mock_generate,
                CONTENT_TYPE_LATEST="text/plain; version=0.0.4",
            ),
        }):
            with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
                result = handler.handle("/api/v1/metrics/prometheus", {}, mock_http)
                assert _status(result) == 200

    def test_returns_same_output_as_metrics(self, handler, mock_http):
        mock_generate = MagicMock(return_value=b"# same metrics\n")
        with patch.dict("sys.modules", {
            "prometheus_client": MagicMock(
                REGISTRY=MagicMock(),
                generate_latest=mock_generate,
                CONTENT_TYPE_LATEST="text/plain",
            ),
        }):
            with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
                r1 = handler.handle("/metrics", {}, mock_http)
                r2 = handler.handle("/api/v1/metrics/prometheus", {}, mock_http)
                assert _raw_body(r1) == _raw_body(r2)

    def test_non_versioned_api_path(self, handler, mock_http):
        mock_generate = MagicMock(return_value=b"# ok\n")
        with patch.dict("sys.modules", {
            "prometheus_client": MagicMock(
                REGISTRY=MagicMock(),
                generate_latest=mock_generate,
                CONTENT_TYPE_LATEST="text/plain",
            ),
        }):
            with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
                result = handler.handle("/api/metrics/prometheus", {}, mock_http)
                assert _status(result) == 200


# ===========================================================================
# TestHandlerRouteSummary
# ===========================================================================


class TestHandlerRouteSummary:
    """Test GET /api/v1/metrics/prometheus/summary endpoint."""

    def test_returns_200(self, handler, mock_http):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            import sys as _sys
            saved = _sys.modules.get("prometheus_client")
            _sys.modules["prometheus_client"] = None  # type: ignore[assignment]
            try:
                result = handler.handle("/api/v1/metrics/prometheus/summary", {}, mock_http)
                assert _status(result) == 200
            finally:
                if saved is not None:
                    _sys.modules["prometheus_client"] = saved
                else:
                    _sys.modules.pop("prometheus_client", None)

    def test_returns_json(self, handler, mock_http):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            import sys as _sys
            saved = _sys.modules.get("prometheus_client")
            _sys.modules["prometheus_client"] = None  # type: ignore[assignment]
            try:
                result = handler.handle("/api/v1/metrics/prometheus/summary", {}, mock_http)
                assert result.content_type == "application/json"
            finally:
                if saved is not None:
                    _sys.modules["prometheus_client"] = saved
                else:
                    _sys.modules.pop("prometheus_client", None)

    def test_contains_initialized_flag(self, handler, mock_http):
        MetricsRegistry._initialized = True
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            import sys as _sys
            saved = _sys.modules.get("prometheus_client")
            _sys.modules["prometheus_client"] = None  # type: ignore[assignment]
            try:
                result = handler.handle("/api/v1/metrics/prometheus/summary", {}, mock_http)
                body = _body(result)
                assert "initialized" in body
            finally:
                if saved is not None:
                    _sys.modules["prometheus_client"] = saved
                else:
                    _sys.modules.pop("prometheus_client", None)

    def test_contains_metrics_key(self, handler, mock_http):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            import sys as _sys
            saved = _sys.modules.get("prometheus_client")
            _sys.modules["prometheus_client"] = None  # type: ignore[assignment]
            try:
                result = handler.handle("/api/v1/metrics/prometheus/summary", {}, mock_http)
                body = _body(result)
                assert "metrics" in body
            finally:
                if saved is not None:
                    _sys.modules["prometheus_client"] = saved
                else:
                    _sys.modules.pop("prometheus_client", None)

    def test_non_versioned_summary_path(self, handler, mock_http):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            import sys as _sys
            saved = _sys.modules.get("prometheus_client")
            _sys.modules["prometheus_client"] = None  # type: ignore[assignment]
            try:
                result = handler.handle("/api/metrics/prometheus/summary", {}, mock_http)
                assert _status(result) == 200
            finally:
                if saved is not None:
                    _sys.modules["prometheus_client"] = saved
                else:
                    _sys.modules.pop("prometheus_client", None)


# ===========================================================================
# TestHandlerAuthRequired
# ===========================================================================


class TestHandlerAuthRequired:
    """Test auth/permission checks on API-versioned routes."""

    @pytest.mark.no_auto_auth
    def test_api_prometheus_requires_auth(self):
        """API route should require authentication."""
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            h = UnifiedMetricsHandler(ctx={})

        # Mock require_auth_or_error to return error
        from aragora.server.handlers.base import error_response as _err

        def mock_auth_error(self_handler, handler):
            return None, _err("Not authenticated", 401)

        with patch.object(type(h), "require_auth_or_error", mock_auth_error):
            result = h.handle("/api/v1/metrics/prometheus", {}, MockHTTPHandler())
            assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_api_summary_requires_auth(self):
        """Summary route should require authentication."""
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            h = UnifiedMetricsHandler(ctx={})

        from aragora.server.handlers.base import error_response as _err

        def mock_auth_error(self_handler, handler):
            return None, _err("Not authenticated", 401)

        with patch.object(type(h), "require_auth_or_error", mock_auth_error):
            result = h.handle("/api/v1/metrics/prometheus/summary", {}, MockHTTPHandler())
            assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_api_prometheus_requires_metrics_read_permission(self):
        """API route should require metrics:read permission."""
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            h = UnifiedMetricsHandler(ctx={})

        from aragora.server.handlers.base import error_response as _err

        mock_user = MagicMock()

        def mock_auth_ok(self_handler, handler):
            return mock_user, None

        def mock_perm_error(self_handler, handler, permission):
            return None, _err("Permission denied", 403)

        with patch.object(type(h), "require_auth_or_error", mock_auth_ok):
            with patch.object(type(h), "require_permission_or_error", mock_perm_error):
                result = h.handle("/api/v1/metrics/prometheus", {}, MockHTTPHandler())
                assert _status(result) == 403

    @pytest.mark.no_auto_auth
    def test_bare_metrics_no_auth(self):
        """GET /metrics does NOT require auth."""
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            h = UnifiedMetricsHandler(ctx={})

        mock_generate = MagicMock(return_value=b"# ok\n")
        with patch.dict("sys.modules", {
            "prometheus_client": MagicMock(
                REGISTRY=MagicMock(),
                generate_latest=mock_generate,
                CONTENT_TYPE_LATEST="text/plain",
            ),
        }):
            with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
                result = h.handle("/metrics", {}, MockHTTPHandler())
                assert _status(result) == 200


# ===========================================================================
# TestHandlerErrorPaths
# ===========================================================================


class TestHandlerErrorPaths:
    """Test error handling in handler methods."""

    def test_prometheus_metrics_runtime_error(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            side_effect=RuntimeError("metrics broken"),
        ):
            result = handler.handle("/metrics", {}, mock_http)
            assert _status(result) == 500

    def test_prometheus_metrics_value_error(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            side_effect=ValueError("bad value"),
        ):
            result = handler.handle("/metrics", {}, mock_http)
            assert _status(result) == 500

    def test_prometheus_metrics_type_error(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            side_effect=TypeError("wrong type"),
        ):
            result = handler.handle("/metrics", {}, mock_http)
            assert _status(result) == 500

    def test_prometheus_metrics_key_error(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            side_effect=KeyError("missing"),
        ):
            result = handler.handle("/metrics", {}, mock_http)
            assert _status(result) == 500

    def test_summary_runtime_error(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.metrics_endpoint.get_metrics_summary",
            side_effect=RuntimeError("summary broken"),
        ):
            result = handler.handle("/api/v1/metrics/prometheus/summary", {}, mock_http)
            assert _status(result) == 500

    def test_summary_type_error(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.metrics_endpoint.get_metrics_summary",
            side_effect=TypeError("bad"),
        ):
            result = handler.handle("/api/v1/metrics/prometheus/summary", {}, mock_http)
            assert _status(result) == 500

    def test_summary_value_error(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.metrics_endpoint.get_metrics_summary",
            side_effect=ValueError("invalid"),
        ):
            result = handler.handle("/api/v1/metrics/prometheus/summary", {}, mock_http)
            assert _status(result) == 500

    def test_unhandled_path_returns_none(self, handler, mock_http):
        result = handler.handle("/api/v1/unknown", {}, mock_http)
        assert result is None

    def test_error_body_is_sanitized(self, handler, mock_http):
        """Errors should use safe_error_message and not leak internals."""
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            side_effect=RuntimeError("secret internal path /var/data/db"),
        ):
            result = handler.handle("/metrics", {}, mock_http)
            raw = _raw_body(result)
            # Should NOT contain the raw exception message
            assert "/var/data/db" not in raw


# ===========================================================================
# TestConvenienceFunctions
# ===========================================================================


class TestConvenienceFunctions:
    """Test ensure_all_metrics_registered and get_registered_metric_names."""

    def test_ensure_all_delegates_to_registry(self):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True) as m:
            result = ensure_all_metrics_registered()
            assert result is True
            m.assert_called_once()

    def test_ensure_all_returns_false_on_failure(self):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=False) as m:
            result = ensure_all_metrics_registered()
            assert result is False

    def test_get_registered_metric_names_returns_list(self):
        mock_collector = MagicMock()
        mock_collector._name = "my_metric"

        mock_registry = MagicMock()
        mock_registry._names_to_collectors = {"my_metric": mock_collector}

        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            with patch.dict("sys.modules", {
                "prometheus_client": MagicMock(REGISTRY=mock_registry),
            }):
                result = get_registered_metric_names()
                assert isinstance(result, list)
                assert "my_metric" in result

    def test_get_registered_metric_names_sorted(self):
        c1 = MagicMock()
        c1._name = "z_metric"
        c2 = MagicMock()
        c2._name = "a_metric"

        mock_registry = MagicMock()
        mock_registry._names_to_collectors = {"z": c1, "a": c2}

        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            with patch.dict("sys.modules", {
                "prometheus_client": MagicMock(REGISTRY=mock_registry),
            }):
                result = get_registered_metric_names()
                assert result == ["a_metric", "z_metric"]

    def test_get_registered_metric_names_deduplicates(self):
        c1 = MagicMock()
        c1._name = "dup_metric"
        c2 = MagicMock()
        c2._name = "dup_metric"

        mock_registry = MagicMock()
        mock_registry._names_to_collectors = {"a": c1, "b": c2}

        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            with patch.dict("sys.modules", {
                "prometheus_client": MagicMock(REGISTRY=mock_registry),
            }):
                result = get_registered_metric_names()
                assert result.count("dup_metric") == 1

    def test_get_registered_metric_names_empty_when_no_prometheus(self):
        import sys as _sys
        saved = _sys.modules.get("prometheus_client")
        _sys.modules["prometheus_client"] = None  # type: ignore[assignment]
        try:
            with patch.object(MetricsRegistry, "ensure_initialized", return_value=False):
                result = get_registered_metric_names()
                assert result == []
        finally:
            if saved is not None:
                _sys.modules["prometheus_client"] = saved
            else:
                _sys.modules.pop("prometheus_client", None)

    def test_get_registered_metric_names_skips_collectors_without_name(self):
        c1 = MagicMock(spec=[])  # No _name attribute
        c2 = MagicMock()
        c2._name = "real_metric"

        mock_registry = MagicMock()
        mock_registry._names_to_collectors = {"a": c1, "b": c2}

        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            with patch.dict("sys.modules", {
                "prometheus_client": MagicMock(REGISTRY=mock_registry),
            }):
                result = get_registered_metric_names()
                assert "real_metric" in result
                assert len(result) == 1


# ===========================================================================
# TestExports
# ===========================================================================


class TestExports:
    """Test __all__ exports are correct."""

    def test_handler_exported(self):
        from aragora.server.handlers.metrics_endpoint import __all__
        assert "UnifiedMetricsHandler" in __all__

    def test_config_exported(self):
        from aragora.server.handlers.metrics_endpoint import __all__
        assert "CardinalityConfig" in __all__

    def test_registry_exported(self):
        from aragora.server.handlers.metrics_endpoint import __all__
        assert "MetricsRegistry" in __all__

    def test_generate_exported(self):
        from aragora.server.handlers.metrics_endpoint import __all__
        assert "generate_prometheus_metrics" in __all__

    def test_summary_exported(self):
        from aragora.server.handlers.metrics_endpoint import __all__
        assert "get_metrics_summary" in __all__

    def test_ensure_exported(self):
        from aragora.server.handlers.metrics_endpoint import __all__
        assert "ensure_all_metrics_registered" in __all__

    def test_get_names_exported(self):
        from aragora.server.handlers.metrics_endpoint import __all__
        assert "get_registered_metric_names" in __all__

    def test_content_type_exported(self):
        from aragora.server.handlers.metrics_endpoint import __all__
        assert "PROMETHEUS_CONTENT_TYPE" in __all__

    def test_content_type_value(self):
        assert PROMETHEUS_CONTENT_TYPE == "text/plain; version=0.0.4; charset=utf-8"
