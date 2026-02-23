"""Tests for unified Prometheus metrics endpoint handler.

Tests the metrics API endpoints including:
- GET /metrics - Full Prometheus-format metrics export
- GET /api/v1/metrics/prometheus - Same as /metrics with API versioning
- GET /api/v1/metrics/prometheus/summary - Aggregated metrics summary

Also tests:
- CardinalityConfig dataclass
- _limit_label_cardinality function
- _normalize_endpoint function
- _normalize_table function
- MetricsRegistry class
- generate_prometheus_metrics function
- _generate_fallback_metrics function
- get_metrics_summary function
- ensure_all_metrics_registered function
- get_registered_metric_names function
"""

import json
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

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

def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _raw_body(result) -> str:
    """Extract raw body text from a HandlerResult."""
    if result is None:
        return ""
    raw = result.body
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return raw


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _content_type(result) -> str:
    """Extract content type from a HandlerResult."""
    if result is None:
        return ""
    return result.content_type


class MockHTTPHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, body: dict[str, Any] | None = None, token: str = "test-valid-token"):
        self.rfile = MagicMock()
        self._body = body
        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {
                "Content-Length": str(len(body_bytes)),
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            }
        else:
            self.rfile.read.return_value = b""
            self.headers = {
                "Content-Length": "0",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
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
    MetricsRegistry._initialized = False
    MetricsRegistry._initialization_time = 0.0
    yield
    MetricsRegistry._initialized = original_initialized
    MetricsRegistry._initialization_time = original_time


@pytest.fixture
def handler():
    """Create a UnifiedMetricsHandler instance with mocked initialization."""
    with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
        return UnifiedMetricsHandler()


@pytest.fixture
def mock_http():
    """Create a mock HTTP handler factory."""
    def _make(body=None, token="test-valid-token"):
        return MockHTTPHandler(body=body, token=token)
    return _make


# ============================================================================
# CardinalityConfig tests
# ============================================================================

class TestCardinalityConfig:
    """Tests for CardinalityConfig dataclass."""

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
        config = CardinalityConfig(high_cardinality_metrics=["custom_metric"])
        assert config.high_cardinality_metrics == ["custom_metric"]

    def test_custom_aggregation_disabled(self):
        config = CardinalityConfig(aggregation_enabled=False)
        assert config.aggregation_enabled is False

    def test_default_high_cardinality_metrics_count(self):
        config = CardinalityConfig()
        assert len(config.high_cardinality_metrics) == 4

    def test_empty_high_cardinality_metrics(self):
        config = CardinalityConfig(high_cardinality_metrics=[])
        assert config.high_cardinality_metrics == []


# ============================================================================
# _normalize_endpoint tests
# ============================================================================

class TestNormalizeEndpoint:
    """Tests for _normalize_endpoint function."""

    def test_replace_uuid(self):
        endpoint = "/api/v1/debates/550e8400-e29b-41d4-a716-446655440000/result"
        result = _normalize_endpoint(endpoint)
        assert ":id" in result
        assert "550e8400" not in result

    def test_replace_uppercase_uuid(self):
        endpoint = "/api/v1/debates/550E8400-E29B-41D4-A716-446655440000/result"
        result = _normalize_endpoint(endpoint)
        assert ":id" in result

    def test_replace_numeric_id(self):
        endpoint = "/api/v1/users/12345/profile"
        result = _normalize_endpoint(endpoint)
        assert "/:id/profile" in result

    def test_replace_base64_token(self):
        # 20+ character base64-like string
        long_token = "A" * 25
        endpoint = f"/api/v1/auth/{long_token}/verify"
        result = _normalize_endpoint(endpoint)
        assert ":token" in result

    def test_no_replacement_needed(self):
        endpoint = "/api/v1/health"
        result = _normalize_endpoint(endpoint)
        assert result == "/api/v1/health"

    def test_multiple_replacements(self):
        endpoint = "/api/v1/users/42/debates/99"
        result = _normalize_endpoint(endpoint)
        assert result == "/api/v1/users/:id/debates/:id"

    def test_empty_endpoint(self):
        result = _normalize_endpoint("")
        assert result == ""

    def test_root_endpoint(self):
        result = _normalize_endpoint("/")
        assert result == "/"

    def test_short_alpha_string_not_replaced(self):
        """Strings under 20 chars should not be replaced as tokens."""
        endpoint = "/api/v1/debates/abcdef/result"
        result = _normalize_endpoint(endpoint)
        assert "abcdef" in result


# ============================================================================
# _normalize_table tests
# ============================================================================

class TestNormalizeTable:
    """Tests for _normalize_table function."""

    def test_replace_sharded_suffix(self):
        result = _normalize_table("events_001")
        assert result == "events_:shard"

    def test_replace_three_digit_shard(self):
        result = _normalize_table("users_123")
        assert result == "users_:shard"

    def test_no_replacement_for_single_digit(self):
        """Single digit suffix is not replaced (less than 2 digits)."""
        result = _normalize_table("table_1")
        assert result == "table_1"

    def test_no_replacement_for_non_numeric(self):
        result = _normalize_table("events_archive")
        assert result == "events_archive"

    def test_empty_table_name(self):
        result = _normalize_table("")
        assert result == ""

    def test_only_digits(self):
        result = _normalize_table("123")
        assert result == "123"

    def test_underscore_two_digits(self):
        result = _normalize_table("logs_42")
        assert result == "logs_:shard"

    def test_multiple_underscores(self):
        """Only the trailing numeric suffix is replaced."""
        result = _normalize_table("my_table_2024_01")
        assert result == "my_table_2024_:shard"


# ============================================================================
# _limit_label_cardinality tests
# ============================================================================

class TestLimitLabelCardinality:
    """Tests for _limit_label_cardinality function."""

    def test_non_high_cardinality_metric_unchanged(self):
        config = CardinalityConfig()
        labels = {"endpoint": "/api/v1/users/123", "method": "GET"}
        result = _limit_label_cardinality("custom_metric", labels, config)
        assert result == labels

    def test_high_cardinality_normalizes_endpoint(self):
        config = CardinalityConfig()
        labels = {"endpoint": "/api/v1/users/42", "method": "GET"}
        result = _limit_label_cardinality(
            "aragora_http_requests_total", labels, config
        )
        assert "/:id" in result["endpoint"]
        assert result["method"] == "GET"

    def test_high_cardinality_normalizes_table(self):
        config = CardinalityConfig()
        labels = {"table": "events_001", "operation": "SELECT"}
        result = _limit_label_cardinality(
            "aragora_db_query_duration_seconds", labels, config
        )
        assert result["table"] == "events_:shard"
        assert result["operation"] == "SELECT"

    def test_original_labels_not_mutated(self):
        config = CardinalityConfig()
        labels = {"endpoint": "/api/v1/users/42"}
        original = labels.copy()
        _limit_label_cardinality("aragora_http_requests_total", labels, config)
        assert labels == original

    def test_no_endpoint_or_table_labels(self):
        config = CardinalityConfig()
        labels = {"method": "GET", "status": "200"}
        result = _limit_label_cardinality(
            "aragora_http_requests_total", labels, config
        )
        assert result == {"method": "GET", "status": "200"}

    def test_both_endpoint_and_table(self):
        config = CardinalityConfig()
        labels = {"endpoint": "/api/v1/users/42", "table": "events_001"}
        result = _limit_label_cardinality(
            "aragora_http_requests_total", labels, config
        )
        assert "/:id" in result["endpoint"]
        assert result["table"] == "events_:shard"

    def test_empty_labels(self):
        config = CardinalityConfig()
        result = _limit_label_cardinality(
            "aragora_http_requests_total", {}, config
        )
        assert result == {}

    def test_custom_high_cardinality_list(self):
        config = CardinalityConfig(high_cardinality_metrics=["my_metric"])
        labels = {"endpoint": "/api/v1/users/42"}
        result = _limit_label_cardinality("my_metric", labels, config)
        assert "/:id" in result["endpoint"]


# ============================================================================
# MetricsRegistry tests
# ============================================================================

class TestMetricsRegistry:
    """Tests for MetricsRegistry class."""

    def test_already_initialized_returns_true(self):
        MetricsRegistry._initialized = True
        assert MetricsRegistry.ensure_initialized() is True

    def test_initialization_records_time(self):
        MetricsRegistry._initialized = False
        with patch(
            "aragora.server.handlers.metrics_endpoint.MetricsRegistry.ensure_initialized",
            wraps=MetricsRegistry.ensure_initialized,
        ):
            # Mock all the imports to succeed
            mock_init = MagicMock(return_value=True)
            with patch.dict("sys.modules", {
                "aragora.observability.metrics": MagicMock(init_core_metrics=mock_init),
                "aragora.observability.metrics.agents": MagicMock(init_agent_provider_metrics=MagicMock()),
                "aragora.observability.metrics.bridge": MagicMock(init_bridge_metrics=MagicMock()),
                "aragora.observability.metrics.km": MagicMock(init_km_metrics=MagicMock()),
                "aragora.observability.metrics.slo": MagicMock(init_slo_metrics=MagicMock()),
                "aragora.observability.metrics.security": MagicMock(init_security_metrics=MagicMock()),
                "aragora.observability.metrics.notification": MagicMock(init_notification_metrics=MagicMock()),
                "aragora.observability.metrics.gauntlet": MagicMock(init_gauntlet_metrics=MagicMock()),
                "aragora.observability.metrics.stores": MagicMock(init_store_metrics=MagicMock()),
                "aragora.observability.metrics.debate": MagicMock(init_debate_metrics=MagicMock()),
                "aragora.observability.metrics.request": MagicMock(init_request_metrics=MagicMock()),
                "aragora.observability.metrics.agent": MagicMock(init_agent_metrics=MagicMock()),
                "aragora.observability.metrics.marketplace": MagicMock(init_marketplace_metrics=MagicMock()),
                "aragora.observability.metrics.explainability": MagicMock(init_explainability_metrics=MagicMock()),
                "aragora.observability.metrics.fabric": MagicMock(init_fabric_metrics=MagicMock()),
                "aragora.observability.metrics.task_queue": MagicMock(init_task_queue_metrics=MagicMock()),
                "aragora.observability.metrics.governance": MagicMock(init_governance_metrics=MagicMock()),
                "aragora.observability.metrics.user_mapping": MagicMock(init_user_mapping_metrics=MagicMock()),
                "aragora.observability.metrics.checkpoint": MagicMock(init_checkpoint_metrics=MagicMock()),
                "aragora.observability.metrics.consensus": MagicMock(
                    init_consensus_metrics=MagicMock(),
                    init_enhanced_consensus_metrics=MagicMock(),
                ),
                "aragora.observability.metrics.tts": MagicMock(init_tts_metrics=MagicMock()),
                "aragora.observability.metrics.cache": MagicMock(init_cache_metrics=MagicMock()),
                "aragora.observability.metrics.convergence": MagicMock(init_convergence_metrics=MagicMock()),
                "aragora.observability.metrics.workflow": MagicMock(init_workflow_metrics=MagicMock()),
                "aragora.observability.metrics.memory": MagicMock(init_memory_metrics=MagicMock()),
                "aragora.observability.metrics.evidence": MagicMock(init_evidence_metrics=MagicMock()),
                "aragora.observability.metrics.ranking": MagicMock(init_ranking_metrics=MagicMock()),
                "aragora.observability.metrics.control_plane": MagicMock(_init_control_plane_metrics=MagicMock()),
                "aragora.observability.metrics.platform": MagicMock(_initialize_platform_metrics=MagicMock()),
                "aragora.observability.metrics.webhook": MagicMock(_init_metrics=MagicMock()),
            }):
                result = MetricsRegistry.ensure_initialized()
                assert result is True
                assert MetricsRegistry._initialized is True

    def test_core_metrics_disabled_returns_false(self):
        mock_init = MagicMock(return_value=False)
        with patch.dict("sys.modules", {
            "aragora.observability.metrics": MagicMock(init_core_metrics=mock_init),
        }):
            result = MetricsRegistry.ensure_initialized()
            assert result is False
            assert MetricsRegistry._initialized is True

    def test_import_error_handled(self):
        """ImportError during initialization is handled gracefully."""
        with patch(
            "builtins.__import__",
            side_effect=ImportError("no module"),
        ):
            MetricsRegistry._initialized = False
            result = MetricsRegistry.ensure_initialized()
            # Should handle the error and set _initialized
            assert MetricsRegistry._initialized is True

    def test_value_error_handled(self):
        """ValueError during initialization is handled gracefully."""
        mock_init = MagicMock(side_effect=ValueError("bad value"))
        with patch.dict("sys.modules", {
            "aragora.observability.metrics": MagicMock(init_core_metrics=mock_init),
        }):
            result = MetricsRegistry.ensure_initialized()
            assert result is False
            assert MetricsRegistry._initialized is True

    def test_runtime_error_handled(self):
        """RuntimeError during initialization is handled gracefully."""
        mock_init = MagicMock(side_effect=RuntimeError("runtime fail"))
        with patch.dict("sys.modules", {
            "aragora.observability.metrics": MagicMock(init_core_metrics=mock_init),
        }):
            result = MetricsRegistry.ensure_initialized()
            assert result is False
            assert MetricsRegistry._initialized is True

    def test_get_initialization_time_default(self):
        MetricsRegistry._initialization_time = 0.0
        assert MetricsRegistry.get_initialization_time() == 0.0

    def test_get_initialization_time_after_init(self):
        MetricsRegistry._initialization_time = 1.234
        assert MetricsRegistry.get_initialization_time() == 1.234


# ============================================================================
# _generate_fallback_metrics tests
# ============================================================================

class TestGenerateFallbackMetrics:
    """Tests for _generate_fallback_metrics function."""

    def test_contains_aragora_info(self):
        result = _generate_fallback_metrics()
        assert "aragora_info" in result

    def test_contains_help_line(self):
        result = _generate_fallback_metrics()
        assert "# HELP aragora_info" in result

    def test_contains_type_line(self):
        result = _generate_fallback_metrics()
        assert "# TYPE aragora_info gauge" in result

    def test_contains_version_unknown(self):
        result = _generate_fallback_metrics()
        assert 'version="unknown"' in result

    def test_contains_prometheus_available_false(self):
        result = _generate_fallback_metrics()
        assert 'prometheus_available="false"' in result

    def test_contains_metrics_initialized(self):
        result = _generate_fallback_metrics()
        assert "aragora_metrics_initialized 0" in result

    def test_returns_string(self):
        result = _generate_fallback_metrics()
        assert isinstance(result, str)

    def test_has_proper_prometheus_format(self):
        result = _generate_fallback_metrics()
        lines = result.strip().split("\n")
        # Should have HELP, TYPE, value for each metric
        assert any(line.startswith("# HELP") for line in lines)
        assert any(line.startswith("# TYPE") for line in lines)


# ============================================================================
# generate_prometheus_metrics tests
# ============================================================================

class TestGeneratePrometheusMetrics:
    """Tests for generate_prometheus_metrics function."""

    def test_returns_tuple_of_string_and_content_type(self):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            mock_registry = MagicMock()
            with patch.dict("sys.modules", {"prometheus_client": MagicMock(
                REGISTRY=mock_registry,
                generate_latest=MagicMock(return_value=b"# metrics\n"),
                CONTENT_TYPE_LATEST="text/plain; version=0.0.4",
            )}):
                content, ct = generate_prometheus_metrics()
                assert isinstance(content, str)
                assert isinstance(ct, str)

    def test_fallback_when_prometheus_not_installed(self):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=False):
            with patch.dict("sys.modules", {"prometheus_client": None}):
                content, ct = generate_prometheus_metrics()
                assert "aragora_info" in content
                assert ct == PROMETHEUS_CONTENT_TYPE

    def test_aggregate_parameter_forwarded(self):
        """aggregate_high_cardinality parameter is accepted."""
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            with patch.dict("sys.modules", {"prometheus_client": MagicMock(
                REGISTRY=MagicMock(),
                generate_latest=MagicMock(return_value=b"# metrics\n"),
                CONTENT_TYPE_LATEST="text/plain; version=0.0.4",
            )}):
                content, ct = generate_prometheus_metrics(aggregate_high_cardinality=True)
                assert isinstance(content, str)


# ============================================================================
# get_metrics_summary tests
# ============================================================================

class TestGetMetricsSummary:
    """Tests for get_metrics_summary function."""

    def test_returns_dict_with_initialized(self):
        MetricsRegistry._initialized = True
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            with patch.dict("sys.modules", {"prometheus_client": None}):
                summary = get_metrics_summary()
                assert "initialized" in summary
                assert summary["initialized"] is True

    def test_returns_initialization_time(self):
        MetricsRegistry._initialized = True
        MetricsRegistry._initialization_time = 0.5
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            with patch.dict("sys.modules", {"prometheus_client": None}):
                summary = get_metrics_summary()
                assert summary["initialization_time_seconds"] == 0.5

    def test_metrics_unavailable_when_prometheus_missing(self):
        MetricsRegistry._initialized = True
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            with patch.dict("sys.modules", {"prometheus_client": None}):
                summary = get_metrics_summary()
                assert summary["metrics"] == {"available": False}

    def test_with_prometheus_available(self):
        MetricsRegistry._initialized = True
        mock_collector = MagicMock()
        mock_collector._type = "counter"
        mock_collector.samples = []

        mock_registry = MagicMock()
        mock_registry.collect.return_value = [mock_collector]

        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            mock_prom = MagicMock()
            mock_prom.REGISTRY = mock_registry
            with patch.dict("sys.modules", {
                "prometheus_client": mock_prom,
                "aragora.observability.metrics.cardinality": None,
            }):
                summary = get_metrics_summary()
                assert "metrics" in summary
                assert summary["metrics"]["counters"] >= 0

    def test_cardinality_unavailable(self):
        MetricsRegistry._initialized = True
        mock_registry = MagicMock()
        mock_registry.collect.return_value = []

        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            mock_prom = MagicMock()
            mock_prom.REGISTRY = mock_registry
            with patch.dict("sys.modules", {
                "prometheus_client": mock_prom,
                "aragora.observability.metrics.cardinality": None,
            }):
                summary = get_metrics_summary()
                assert summary["cardinality"] == {"available": False}


# ============================================================================
# ensure_all_metrics_registered tests
# ============================================================================

class TestEnsureAllMetricsRegistered:
    """Tests for ensure_all_metrics_registered convenience function."""

    def test_delegates_to_registry(self):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True) as mock_init:
            result = ensure_all_metrics_registered()
            assert result is True
            mock_init.assert_called_once()

    def test_returns_false_when_disabled(self):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=False):
            assert ensure_all_metrics_registered() is False


# ============================================================================
# get_registered_metric_names tests
# ============================================================================

class TestGetRegisteredMetricNames:
    """Tests for get_registered_metric_names function."""

    def test_returns_empty_list_when_prometheus_missing(self):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=False):
            with patch.dict("sys.modules", {"prometheus_client": None}):
                result = get_registered_metric_names()
                assert result == []

    def test_returns_sorted_unique_names(self):
        mock_collector_a = MagicMock()
        mock_collector_a._name = "metric_b"
        mock_collector_b = MagicMock()
        mock_collector_b._name = "metric_a"
        mock_collector_c = MagicMock()
        mock_collector_c._name = "metric_b"  # Duplicate

        mock_registry = MagicMock()
        mock_registry._names_to_collectors = {
            "b": mock_collector_a,
            "a": mock_collector_b,
            "c": mock_collector_c,
        }

        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            mock_prom = MagicMock()
            mock_prom.REGISTRY = mock_registry
            with patch.dict("sys.modules", {"prometheus_client": mock_prom}):
                result = get_registered_metric_names()
                assert result == ["metric_a", "metric_b"]

    def test_skips_collectors_without_name(self):
        mock_collector = MagicMock(spec=[])
        mock_registry = MagicMock()
        mock_registry._names_to_collectors = {"x": mock_collector}

        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            mock_prom = MagicMock()
            mock_prom.REGISTRY = mock_registry
            with patch.dict("sys.modules", {"prometheus_client": mock_prom}):
                result = get_registered_metric_names()
                assert result == []


# ============================================================================
# PROMETHEUS_CONTENT_TYPE constant
# ============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_prometheus_content_type(self):
        assert PROMETHEUS_CONTENT_TYPE == "text/plain; version=0.0.4; charset=utf-8"


# ============================================================================
# UnifiedMetricsHandler initialization tests
# ============================================================================

class TestHandlerInit:
    """Tests for UnifiedMetricsHandler initialization."""

    def test_init_with_ctx(self):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            h = UnifiedMetricsHandler(ctx={"key": "value"})
            assert h.ctx == {"key": "value"}

    def test_init_with_none_ctx(self):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            h = UnifiedMetricsHandler(ctx=None)
            assert h.ctx == {}

    def test_init_default_ctx(self):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            h = UnifiedMetricsHandler()
            assert h.ctx == {}

    def test_init_creates_cardinality_config(self):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            h = UnifiedMetricsHandler()
            assert isinstance(h._cardinality_config, CardinalityConfig)

    def test_init_calls_ensure_initialized(self):
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True) as mock_init:
            UnifiedMetricsHandler()
            mock_init.assert_called_once()


# ============================================================================
# can_handle tests
# ============================================================================

class TestCanHandle:
    """Tests for UnifiedMetricsHandler.can_handle()."""

    def test_metrics_root(self, handler):
        assert handler.can_handle("/metrics") is True

    def test_api_metrics_prometheus(self, handler):
        assert handler.can_handle("/api/v1/metrics/prometheus") is True

    def test_api_metrics_prometheus_summary(self, handler):
        assert handler.can_handle("/api/v1/metrics/prometheus/summary") is True

    def test_api_v2_metrics_prometheus(self, handler):
        """Version stripping should work for v2 as well."""
        assert handler.can_handle("/api/v2/metrics/prometheus") is True

    def test_unknown_path(self, handler):
        assert handler.can_handle("/api/v1/health") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_partial_metrics_path(self, handler):
        assert handler.can_handle("/api/v1/metrics") is False

    def test_metrics_with_extra_suffix(self, handler):
        assert handler.can_handle("/api/v1/metrics/prometheus/summary/extra") is False

    def test_root_slash(self, handler):
        assert handler.can_handle("/") is False

    def test_without_version_prefix(self, handler):
        """Path without version prefix should also match."""
        assert handler.can_handle("/api/metrics/prometheus") is True

    def test_without_version_prefix_summary(self, handler):
        assert handler.can_handle("/api/metrics/prometheus/summary") is True


# ============================================================================
# ROUTES constant tests
# ============================================================================

class TestRoutes:
    """Tests for the ROUTES class attribute."""

    def test_routes_contains_metrics(self, handler):
        assert "/metrics" in handler.ROUTES

    def test_routes_contains_api_prometheus(self, handler):
        assert "/api/metrics/prometheus" in handler.ROUTES

    def test_routes_contains_api_prometheus_summary(self, handler):
        assert "/api/metrics/prometheus/summary" in handler.ROUTES

    def test_routes_count(self, handler):
        assert len(handler.ROUTES) == 3


# ============================================================================
# handle() routing tests - GET /metrics
# ============================================================================

class TestHandleMetricsRoute:
    """Tests for handle() routing to GET /metrics."""

    def test_metrics_returns_200(self, handler, mock_http):
        with patch.object(handler, "_get_prometheus_metrics") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, body=b"# metrics")
            result = handler.handle("/metrics", {}, mock_http())
            assert _status(result) == 200

    def test_metrics_dispatches_to_get_prometheus_metrics(self, handler, mock_http):
        with patch.object(handler, "_get_prometheus_metrics") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, body=b"# metrics")
            handler.handle("/metrics", {}, mock_http())
            mock_get.assert_called_once_with({})

    def test_metrics_passes_query_params(self, handler, mock_http):
        with patch.object(handler, "_get_prometheus_metrics") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, body=b"# metrics")
            params = {"aggregate": ["true"]}
            handler.handle("/metrics", params, mock_http())
            mock_get.assert_called_once_with(params)

    def test_metrics_no_auth_required(self, handler, mock_http):
        """The /metrics path does NOT require auth (no require_auth_or_error call)."""
        with patch.object(handler, "_get_prometheus_metrics") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, body=b"# metrics")
            with patch.object(handler, "require_auth_or_error") as mock_auth:
                handler.handle("/metrics", {}, mock_http())
                mock_auth.assert_not_called()


# ============================================================================
# handle() routing tests - GET /api/v1/metrics/prometheus
# ============================================================================

class TestHandleApiPrometheusRoute:
    """Tests for handle() routing to GET /api/v1/metrics/prometheus."""

    def test_api_prometheus_returns_200(self, handler, mock_http):
        with patch.object(handler, "_get_prometheus_metrics") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, body=b"# metrics")
            result = handler.handle("/api/v1/metrics/prometheus", {}, mock_http())
            assert _status(result) == 200

    def test_api_prometheus_dispatches_correctly(self, handler, mock_http):
        with patch.object(handler, "_get_prometheus_metrics") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, body=b"# metrics")
            handler.handle("/api/v1/metrics/prometheus", {}, mock_http())
            mock_get.assert_called_once()

    def test_api_prometheus_requires_auth(self, handler, mock_http):
        """The API-versioned path requires authentication."""
        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (MagicMock(), None)
            with patch.object(handler, "require_permission_or_error") as mock_perm:
                mock_perm.return_value = (MagicMock(), None)
                with patch.object(handler, "_get_prometheus_metrics") as mock_get:
                    mock_get.return_value = MagicMock(status_code=200, body=b"# metrics")
                    handler.handle("/api/v1/metrics/prometheus", {}, mock_http())
                    mock_auth.assert_called_once()

    def test_api_prometheus_requires_metrics_read_permission(self, handler, mock_http):
        """The API-versioned path requires metrics:read permission."""
        http = mock_http()
        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (MagicMock(), None)
            with patch.object(handler, "require_permission_or_error") as mock_perm:
                mock_perm.return_value = (MagicMock(), None)
                with patch.object(handler, "_get_prometheus_metrics") as mock_get:
                    mock_get.return_value = MagicMock(status_code=200, body=b"# metrics")
                    handler.handle("/api/v1/metrics/prometheus", {}, http)
                    mock_perm.assert_called_once_with(http, "metrics:read")


# ============================================================================
# handle() routing tests - GET /api/v1/metrics/prometheus/summary
# ============================================================================

class TestHandleApiSummaryRoute:
    """Tests for handle() routing to GET /api/v1/metrics/prometheus/summary."""

    def test_summary_returns_200(self, handler, mock_http):
        with patch.object(handler, "_get_metrics_summary") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, body=b'{}')
            result = handler.handle("/api/v1/metrics/prometheus/summary", {}, mock_http())
            assert _status(result) == 200

    def test_summary_dispatches_correctly(self, handler, mock_http):
        with patch.object(handler, "_get_metrics_summary") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, body=b'{}')
            handler.handle("/api/v1/metrics/prometheus/summary", {}, mock_http())
            mock_get.assert_called_once()

    def test_summary_requires_auth(self, handler, mock_http):
        """The summary path requires authentication."""
        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (MagicMock(), None)
            with patch.object(handler, "require_permission_or_error") as mock_perm:
                mock_perm.return_value = (MagicMock(), None)
                with patch.object(handler, "_get_metrics_summary") as mock_get:
                    mock_get.return_value = MagicMock(status_code=200, body=b'{}')
                    handler.handle("/api/v1/metrics/prometheus/summary", {}, mock_http())
                    mock_auth.assert_called_once()


# ============================================================================
# handle() - auth failure tests
# ============================================================================

class TestHandleAuthFailure:
    """Tests for auth failures on protected endpoints."""

    @pytest.mark.no_auto_auth
    def test_auth_failure_returns_error_on_api_prometheus(self, mock_http):
        """Auth failure on /api/v1/metrics/prometheus returns error."""
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            h = UnifiedMetricsHandler()

        error_result = MagicMock(status_code=401, body=b'{"error":"Unauthorized"}')
        with patch.object(h, "require_auth_or_error", return_value=(None, error_result)):
            result = h.handle("/api/v1/metrics/prometheus", {}, mock_http())
            assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_permission_failure_returns_error(self, mock_http):
        """Permission failure on /api/v1/metrics/prometheus returns error."""
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            h = UnifiedMetricsHandler()

        perm_error = MagicMock(status_code=403, body=b'{"error":"Forbidden"}')
        with patch.object(h, "require_auth_or_error", return_value=(MagicMock(), None)):
            with patch.object(h, "require_permission_or_error", return_value=(None, perm_error)):
                result = h.handle("/api/v1/metrics/prometheus", {}, mock_http())
                assert _status(result) == 403

    @pytest.mark.no_auto_auth
    def test_auth_failure_on_summary_returns_error(self, mock_http):
        """Auth failure on summary endpoint returns error."""
        with patch.object(MetricsRegistry, "ensure_initialized", return_value=True):
            h = UnifiedMetricsHandler()

        error_result = MagicMock(status_code=401, body=b'{"error":"Unauthorized"}')
        with patch.object(h, "require_auth_or_error", return_value=(None, error_result)):
            result = h.handle("/api/v1/metrics/prometheus/summary", {}, mock_http())
            assert _status(result) == 401


# ============================================================================
# handle() - unmatched routes
# ============================================================================

class TestHandleUnmatchedRoute:
    """Tests for handle() returning None for unrecognized paths."""

    def test_returns_none_for_unknown_path(self, handler, mock_http):
        result = handler.handle("/api/v1/unknown", {}, mock_http())
        assert result is None

    def test_returns_none_for_empty_path(self, handler, mock_http):
        result = handler.handle("", {}, mock_http())
        assert result is None

    def test_returns_none_for_root(self, handler, mock_http):
        result = handler.handle("/", {}, mock_http())
        assert result is None

    def test_returns_none_for_partial_metrics(self, handler, mock_http):
        result = handler.handle("/api/v1/metrics", {}, mock_http())
        assert result is None


# ============================================================================
# _get_prometheus_metrics tests
# ============================================================================

class TestGetPrometheusMetrics:
    """Tests for _get_prometheus_metrics handler method."""

    def test_success_returns_200(self, handler):
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            return_value=("# metrics data\n", "text/plain; version=0.0.4"),
        ):
            result = handler._get_prometheus_metrics({})
            assert _status(result) == 200

    def test_success_sets_content_type(self, handler):
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            return_value=("# metrics data\n", "text/plain; version=0.0.4"),
        ):
            result = handler._get_prometheus_metrics({})
            assert _content_type(result) == "text/plain; version=0.0.4"

    def test_success_body_is_bytes(self, handler):
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            return_value=("# metrics data\n", "text/plain; version=0.0.4"),
        ):
            result = handler._get_prometheus_metrics({})
            assert isinstance(result.body, bytes)

    def test_aggregate_param_true(self, handler):
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            return_value=("# metrics\n", "text/plain"),
        ) as mock_gen:
            handler._get_prometheus_metrics({"aggregate": ["true"]})
            mock_gen.assert_called_once_with(aggregate_high_cardinality=True)

    def test_aggregate_param_false(self, handler):
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            return_value=("# metrics\n", "text/plain"),
        ) as mock_gen:
            handler._get_prometheus_metrics({"aggregate": ["false"]})
            mock_gen.assert_called_once_with(aggregate_high_cardinality=False)

    def test_aggregate_param_missing_defaults_false(self, handler):
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            return_value=("# metrics\n", "text/plain"),
        ) as mock_gen:
            handler._get_prometheus_metrics({})
            mock_gen.assert_called_once_with(aggregate_high_cardinality=False)

    def test_aggregate_param_case_insensitive(self, handler):
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            return_value=("# metrics\n", "text/plain"),
        ) as mock_gen:
            handler._get_prometheus_metrics({"aggregate": ["TRUE"]})
            mock_gen.assert_called_once_with(aggregate_high_cardinality=True)

    def test_key_error_returns_500(self, handler):
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            side_effect=KeyError("missing_key"),
        ):
            result = handler._get_prometheus_metrics({})
            assert _status(result) == 500

    def test_value_error_returns_500(self, handler):
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            side_effect=ValueError("bad value"),
        ):
            result = handler._get_prometheus_metrics({})
            assert _status(result) == 500

    def test_type_error_returns_500(self, handler):
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            side_effect=TypeError("type mismatch"),
        ):
            result = handler._get_prometheus_metrics({})
            assert _status(result) == 500

    def test_runtime_error_returns_500(self, handler):
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            side_effect=RuntimeError("runtime fail"),
        ):
            result = handler._get_prometheus_metrics({})
            assert _status(result) == 500

    def test_error_body_has_error_key(self, handler):
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            side_effect=RuntimeError("runtime fail"),
        ):
            result = handler._get_prometheus_metrics({})
            body = _body(result)
            assert "error" in body

    def test_body_content_is_metrics_text(self, handler):
        metrics_text = "# HELP my_metric A metric\n# TYPE my_metric gauge\nmy_metric 42\n"
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            return_value=(metrics_text, "text/plain"),
        ):
            result = handler._get_prometheus_metrics({})
            assert _raw_body(result) == metrics_text


# ============================================================================
# _get_metrics_summary tests
# ============================================================================

class TestGetMetricsSummaryHandler:
    """Tests for _get_metrics_summary handler method."""

    def test_success_returns_200(self, handler):
        with patch(
            "aragora.server.handlers.metrics_endpoint.get_metrics_summary",
            return_value={"initialized": True, "metrics": {}},
        ):
            result = handler._get_metrics_summary()
            assert _status(result) == 200

    def test_success_content_type_json(self, handler):
        with patch(
            "aragora.server.handlers.metrics_endpoint.get_metrics_summary",
            return_value={"initialized": True, "metrics": {}},
        ):
            result = handler._get_metrics_summary()
            assert _content_type(result) == "application/json"

    def test_success_body_is_json(self, handler):
        with patch(
            "aragora.server.handlers.metrics_endpoint.get_metrics_summary",
            return_value={"initialized": True, "initialization_time_seconds": 0.1, "metrics": {}},
        ):
            result = handler._get_metrics_summary()
            body = _body(result)
            assert body["initialized"] is True
            assert body["initialization_time_seconds"] == 0.1

    def test_error_returns_500(self, handler):
        with patch(
            "aragora.server.handlers.metrics_endpoint.get_metrics_summary",
            side_effect=RuntimeError("fail"),
        ):
            result = handler._get_metrics_summary()
            assert _status(result) == 500

    def test_type_error_returns_500(self, handler):
        with patch(
            "aragora.server.handlers.metrics_endpoint.get_metrics_summary",
            side_effect=TypeError("bad type"),
        ):
            result = handler._get_metrics_summary()
            assert _status(result) == 500

    def test_value_error_returns_500(self, handler):
        with patch(
            "aragora.server.handlers.metrics_endpoint.get_metrics_summary",
            side_effect=ValueError("bad value"),
        ):
            result = handler._get_metrics_summary()
            assert _status(result) == 500

    def test_import_error_returns_500(self, handler):
        with patch(
            "aragora.server.handlers.metrics_endpoint.get_metrics_summary",
            side_effect=ImportError("no json"),
        ):
            result = handler._get_metrics_summary()
            assert _status(result) == 500

    def test_body_is_bytes(self, handler):
        with patch(
            "aragora.server.handlers.metrics_endpoint.get_metrics_summary",
            return_value={"initialized": True, "metrics": {}},
        ):
            result = handler._get_metrics_summary()
            assert isinstance(result.body, bytes)

    def test_body_is_indented_json(self, handler):
        with patch(
            "aragora.server.handlers.metrics_endpoint.get_metrics_summary",
            return_value={"initialized": True, "metrics": {}},
        ):
            result = handler._get_metrics_summary()
            raw = result.body.decode("utf-8")
            # Indented JSON should have newlines
            assert "\n" in raw

    def test_error_body_has_error_key(self, handler):
        with patch(
            "aragora.server.handlers.metrics_endpoint.get_metrics_summary",
            side_effect=RuntimeError("fail"),
        ):
            result = handler._get_metrics_summary()
            body = _body(result)
            assert "error" in body


# ============================================================================
# Integration-style tests (full path through handle)
# ============================================================================

class TestIntegration:
    """Integration-style tests exercising full handle path."""

    def test_full_metrics_path(self, handler, mock_http):
        """Full path through /metrics with mocked generation."""
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            return_value=("# test metrics\n", "text/plain; version=0.0.4"),
        ):
            result = handler.handle("/metrics", {}, mock_http())
            assert _status(result) == 200
            assert "test metrics" in _raw_body(result)

    def test_full_api_prometheus_path(self, handler, mock_http):
        """Full path through /api/v1/metrics/prometheus."""
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            return_value=("# api metrics\n", "text/plain; version=0.0.4"),
        ):
            result = handler.handle("/api/v1/metrics/prometheus", {}, mock_http())
            assert _status(result) == 200
            assert "api metrics" in _raw_body(result)

    def test_full_summary_path(self, handler, mock_http):
        """Full path through /api/v1/metrics/prometheus/summary."""
        with patch(
            "aragora.server.handlers.metrics_endpoint.get_metrics_summary",
            return_value={"initialized": True, "initialization_time_seconds": 0.5, "metrics": {"counters": 5}},
        ):
            result = handler.handle("/api/v1/metrics/prometheus/summary", {}, mock_http())
            assert _status(result) == 200
            body = _body(result)
            assert body["metrics"]["counters"] == 5

    def test_metrics_with_aggregate_true(self, handler, mock_http):
        """Full path with aggregate=true query param."""
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            return_value=("# aggregated\n", "text/plain"),
        ) as mock_gen:
            result = handler.handle("/metrics", {"aggregate": ["true"]}, mock_http())
            assert _status(result) == 200
            mock_gen.assert_called_once_with(aggregate_high_cardinality=True)

    def test_metrics_with_aggregate_false(self, handler, mock_http):
        """Full path with aggregate=false query param."""
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            return_value=("# raw\n", "text/plain"),
        ) as mock_gen:
            result = handler.handle("/metrics", {"aggregate": ["false"]}, mock_http())
            assert _status(result) == 200
            mock_gen.assert_called_once_with(aggregate_high_cardinality=False)

    def test_error_during_generation(self, handler, mock_http):
        """Error during metrics generation returns 500."""
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            side_effect=RuntimeError("generation failed"),
        ):
            result = handler.handle("/metrics", {}, mock_http())
            assert _status(result) == 500

    def test_error_during_summary(self, handler, mock_http):
        """Error during summary returns 500."""
        with patch(
            "aragora.server.handlers.metrics_endpoint.get_metrics_summary",
            side_effect=TypeError("summary failed"),
        ):
            result = handler.handle("/api/v1/metrics/prometheus/summary", {}, mock_http())
            assert _status(result) == 500


# ============================================================================
# Edge case and additional coverage tests
# ============================================================================

class TestEdgeCases:
    """Edge case tests for comprehensive coverage."""

    def test_normalize_endpoint_multiple_uuids(self):
        endpoint = "/api/550e8400-e29b-41d4-a716-446655440000/sub/660e8400-e29b-41d4-a716-446655440001"
        result = _normalize_endpoint(endpoint)
        # Both UUIDs should be replaced
        assert result.count(":id") == 2

    def test_normalize_endpoint_mixed_ids(self):
        """Endpoint with both UUID and numeric ID."""
        endpoint = "/api/550e8400-e29b-41d4-a716-446655440000/items/42"
        result = _normalize_endpoint(endpoint)
        assert ":id" in result
        assert "550e8400" not in result
        assert "/42" not in result

    def test_normalize_table_no_underscore(self):
        result = _normalize_table("events")
        assert result == "events"

    def test_cardinality_config_instances_independent(self):
        """Each CardinalityConfig instance has its own list."""
        c1 = CardinalityConfig()
        c2 = CardinalityConfig()
        c1.high_cardinality_metrics.append("new_metric")
        assert "new_metric" not in c2.high_cardinality_metrics

    def test_handler_handle_versioned_v2(self, handler, mock_http):
        """v2 path should also work after version stripping."""
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            return_value=("# v2 metrics\n", "text/plain"),
        ):
            result = handler.handle("/api/v2/metrics/prometheus", {}, mock_http())
            assert _status(result) == 200

    def test_generate_fallback_metrics_is_parseable(self):
        """Fallback metrics should be parseable Prometheus text format."""
        result = _generate_fallback_metrics()
        lines = result.strip().split("\n")
        for line in lines:
            if not line:
                continue
            # Lines should be comments or metric values
            assert line.startswith("#") or line[0].isalpha()

    def test_aggregate_param_with_mixed_case(self, handler):
        """Aggregate with mixed case 'True' should resolve to true."""
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            return_value=("# metrics\n", "text/plain"),
        ) as mock_gen:
            handler._get_prometheus_metrics({"aggregate": ["True"]})
            mock_gen.assert_called_once_with(aggregate_high_cardinality=True)

    def test_aggregate_param_arbitrary_string(self, handler):
        """Arbitrary string for aggregate should resolve to false."""
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            return_value=("# metrics\n", "text/plain"),
        ) as mock_gen:
            handler._get_prometheus_metrics({"aggregate": ["yes"]})
            mock_gen.assert_called_once_with(aggregate_high_cardinality=False)

    def test_handler_exports(self):
        """Verify __all__ exports match expected."""
        from aragora.server.handlers.metrics_endpoint import __all__
        assert "UnifiedMetricsHandler" in __all__
        assert "CardinalityConfig" in __all__
        assert "MetricsRegistry" in __all__
        assert "generate_prometheus_metrics" in __all__
        assert "get_metrics_summary" in __all__
        assert "ensure_all_metrics_registered" in __all__
        assert "get_registered_metric_names" in __all__
        assert "PROMETHEUS_CONTENT_TYPE" in __all__

    def test_metrics_endpoint_body_encoding(self, handler):
        """Body should be UTF-8 encoded."""
        unicode_metrics = "# HELP my_metric A m\u00e9tric\nmy_metric 1\n"
        with patch(
            "aragora.server.handlers.metrics_endpoint.generate_prometheus_metrics",
            return_value=(unicode_metrics, "text/plain"),
        ):
            result = handler._get_prometheus_metrics({})
            assert result.body == unicode_metrics.encode("utf-8")

    def test_normalize_endpoint_preserves_static_paths(self):
        """Static endpoint paths should not be modified."""
        endpoint = "/api/v1/health/ready"
        result = _normalize_endpoint(endpoint)
        assert result == "/api/v1/health/ready"

    def test_normalize_endpoint_base64_exactly_20_chars(self):
        """Token at exactly 20 chars should be replaced."""
        token = "A" * 20
        endpoint = f"/api/{token}/check"
        result = _normalize_endpoint(endpoint)
        assert ":token" in result

    def test_normalize_endpoint_base64_19_chars_not_replaced(self):
        """Token at 19 chars should NOT be replaced."""
        token = "A" * 19
        endpoint = f"/api/{token}/check"
        result = _normalize_endpoint(endpoint)
        assert token in result
