"""
Tests for observability configuration.

Tests cover:
- TracingConfig dataclass defaults and validation
- MetricsConfig dataclass defaults
- get_tracing_config environment variable loading and precedence
- get_metrics_config environment variable loading
- set_tracing_config / set_metrics_config overrides
- reset_config clearing cached singletons
- is_tracing_enabled / is_metrics_enabled convenience helpers
"""

import pytest
from unittest.mock import patch

from aragora.observability.config import (
    TracingConfig,
    MetricsConfig,
    get_tracing_config,
    get_metrics_config,
    set_tracing_config,
    set_metrics_config,
    reset_config,
    is_tracing_enabled,
    is_metrics_enabled,
)


@pytest.fixture(autouse=True)
def _reset():
    """Reset global config singletons before each test."""
    reset_config()
    yield
    reset_config()


# --- TracingConfig dataclass tests ---


class TestTracingConfigDefaults:
    """Tests for TracingConfig default values and validation."""

    def test_defaults(self):
        cfg = TracingConfig()
        assert cfg.enabled is False
        assert cfg.otlp_endpoint == "http://localhost:4317"
        assert cfg.service_name == "aragora"
        assert cfg.service_version == "1.0.0"
        assert cfg.environment == "development"
        assert cfg.sample_rate == 1.0
        assert cfg.sampler_type == "parentbased_traceidratio"
        assert cfg.propagators == ["tracecontext", "baggage"]
        assert cfg.batch_size == 512
        assert cfg.export_timeout_ms == 30000
        assert cfg.insecure is False

    def test_custom_values(self):
        cfg = TracingConfig(
            enabled=True,
            otlp_endpoint="http://collector:4318",
            service_name="my-svc",
            sample_rate=0.5,
            batch_size=1024,
        )
        assert cfg.enabled is True
        assert cfg.otlp_endpoint == "http://collector:4318"
        assert cfg.service_name == "my-svc"
        assert cfg.sample_rate == 0.5
        assert cfg.batch_size == 1024

    def test_sample_rate_too_high(self):
        with pytest.raises(ValueError, match="sample_rate must be between"):
            TracingConfig(sample_rate=1.5)

    def test_sample_rate_negative(self):
        with pytest.raises(ValueError, match="sample_rate must be between"):
            TracingConfig(sample_rate=-0.1)

    def test_batch_size_zero(self):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TracingConfig(batch_size=0)

    def test_batch_size_negative(self):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TracingConfig(batch_size=-10)

    def test_sample_rate_boundary_zero(self):
        cfg = TracingConfig(sample_rate=0.0)
        assert cfg.sample_rate == 0.0

    def test_sample_rate_boundary_one(self):
        cfg = TracingConfig(sample_rate=1.0)
        assert cfg.sample_rate == 1.0


# --- MetricsConfig dataclass tests ---


class TestMetricsConfigDefaults:
    """Tests for MetricsConfig default values."""

    def test_defaults(self):
        cfg = MetricsConfig()
        assert cfg.enabled is True
        assert cfg.port == 9090
        assert cfg.path == "/metrics"
        assert cfg.include_host_metrics is False
        assert len(cfg.histogram_buckets) == 11


# --- get_tracing_config environment variable loading ---


class TestGetTracingConfig:
    """Tests for get_tracing_config loading from environment."""

    @patch.dict("os.environ", {}, clear=True)
    def test_defaults_no_env(self):
        cfg = get_tracing_config()
        assert cfg.enabled is False
        assert cfg.otlp_endpoint == "http://localhost:4317"
        assert cfg.service_name == "aragora"

    @patch.dict(
        "os.environ",
        {"OTEL_ENABLED": "true", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel:4317"},
        clear=True,
    )
    def test_otel_enabled_explicitly(self):
        cfg = get_tracing_config()
        assert cfg.enabled is True
        assert cfg.otlp_endpoint == "http://otel:4317"

    @patch.dict(
        "os.environ",
        {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://auto:4317"},
        clear=True,
    )
    def test_auto_enable_via_otel_endpoint(self):
        cfg = get_tracing_config()
        assert cfg.enabled is True

    @patch.dict(
        "os.environ",
        {"ARAGORA_OTLP_EXPORTER": "jaeger"},
        clear=True,
    )
    def test_auto_enable_via_aragora_exporter(self):
        cfg = get_tracing_config()
        assert cfg.enabled is True

    @patch.dict(
        "os.environ",
        {
            "OTEL_SERVICE_NAME": "otel-name",
            "ARAGORA_SERVICE_NAME": "aragora-name",
            "OTEL_ENABLED": "1",
        },
        clear=True,
    )
    def test_otel_service_name_takes_precedence(self):
        cfg = get_tracing_config()
        assert cfg.service_name == "otel-name"

    @patch.dict(
        "os.environ",
        {"ARAGORA_SERVICE_NAME": "aragora-name", "OTEL_ENABLED": "yes"},
        clear=True,
    )
    def test_aragora_service_name_fallback(self):
        cfg = get_tracing_config()
        assert cfg.service_name == "aragora-name"

    @patch.dict(
        "os.environ",
        {
            "OTEL_TRACES_SAMPLER_ARG": "0.25",
            "OTEL_SAMPLE_RATE": "0.5",
            "ARAGORA_TRACE_SAMPLE_RATE": "0.75",
        },
        clear=True,
    )
    def test_sample_rate_precedence(self):
        cfg = get_tracing_config()
        assert cfg.sample_rate == 0.25

    @patch.dict(
        "os.environ",
        {"OTEL_PROPAGATORS": "b3,jaeger", "OTEL_ENABLED": "true"},
        clear=True,
    )
    def test_custom_propagators(self):
        cfg = get_tracing_config()
        assert cfg.propagators == ["b3", "jaeger"]

    @patch.dict(
        "os.environ",
        {"ARAGORA_OTLP_INSECURE": "true"},
        clear=True,
    )
    def test_insecure_mode(self):
        cfg = get_tracing_config()
        assert cfg.insecure is True

    def test_caching(self):
        """get_tracing_config returns the same cached instance."""
        cfg1 = get_tracing_config()
        cfg2 = get_tracing_config()
        assert cfg1 is cfg2


# --- get_metrics_config ---


class TestGetMetricsConfig:
    """Tests for get_metrics_config loading from environment."""

    @patch.dict("os.environ", {}, clear=True)
    def test_defaults(self):
        cfg = get_metrics_config()
        assert cfg.enabled is True
        assert cfg.port == 9090

    @patch.dict(
        "os.environ",
        {"METRICS_ENABLED": "false", "METRICS_PORT": "8888"},
        clear=True,
    )
    def test_custom_env(self):
        cfg = get_metrics_config()
        assert cfg.enabled is False
        assert cfg.port == 8888


# --- set / reset helpers ---


class TestConfigHelpers:
    """Tests for set_*_config, reset_config, and convenience checkers."""

    def test_set_tracing_config(self):
        custom = TracingConfig(enabled=True, service_name="custom")
        set_tracing_config(custom)
        assert get_tracing_config() is custom

    def test_set_metrics_config(self):
        custom = MetricsConfig(enabled=False, port=1234)
        set_metrics_config(custom)
        assert get_metrics_config() is custom

    def test_is_tracing_enabled_false_by_default(self):
        assert is_tracing_enabled() is False

    def test_is_tracing_enabled_true(self):
        set_tracing_config(TracingConfig(enabled=True))
        assert is_tracing_enabled() is True

    def test_is_metrics_enabled_true_by_default(self):
        assert is_metrics_enabled() is True

    def test_is_metrics_enabled_false(self):
        set_metrics_config(MetricsConfig(enabled=False))
        assert is_metrics_enabled() is False
