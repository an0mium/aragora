"""Tests for aragora.telemetry module exports and basic instantiation."""

import importlib
import logging

import pytest


# ---------------------------------------------------------------------------
# Import verification
# ---------------------------------------------------------------------------


class TestModuleImport:
    """Verify the telemetry module can be imported."""

    def test_import_telemetry_module(self):
        mod = importlib.import_module("aragora.telemetry")
        assert mod is not None

    def test_import_observability_module(self):
        """Telemetry delegates to observability; ensure it loads too."""
        mod = importlib.import_module("aragora.observability")
        assert mod is not None


# ---------------------------------------------------------------------------
# Key exports exist
# ---------------------------------------------------------------------------

# All names listed in aragora.telemetry.__all__
EXPECTED_EXPORTS = [
    # Logging
    "configure_logging",
    "get_logger",
    "StructuredLogger",
    "LogConfig",
    "set_correlation_id",
    "get_correlation_id",
    "correlation_context",
    # Tracing
    "get_tracer",
    "create_span",
    "trace_handler",
    "trace_async_handler",
    "trace_agent_call",
    "trace_debate",
    "trace_debate_phase",
    "trace_consensus_check",
    "trace_memory_operation",
    "add_span_attributes",
    "record_exception",
    "shutdown_tracing",
    # Metrics
    "start_metrics_server",
    "record_request",
    "record_agent_call",
    "track_debate",
    "set_consensus_rate",
    "record_memory_operation",
    "track_websocket_connection",
    "measure_latency",
    "measure_async_latency",
    "record_debate_completion",
    "record_phase_duration",
    "record_agent_participation",
    "track_phase",
    "record_cache_hit",
    "record_cache_miss",
    # Configuration
    "TracingConfig",
    "MetricsConfig",
    "get_tracing_config",
    "get_metrics_config",
    "is_tracing_enabled",
    "is_metrics_enabled",
]


class TestExportsExist:
    """Verify every name in __all__ is accessible on the module."""

    @pytest.mark.parametrize("name", EXPECTED_EXPORTS)
    def test_export_exists(self, name: str):
        from aragora import telemetry

        assert hasattr(telemetry, name), f"Missing export: {name}"

    def test_all_matches_expected(self):
        """__all__ should contain at least the expected exports."""
        from aragora import telemetry

        all_set = set(telemetry.__all__)
        for name in EXPECTED_EXPORTS:
            assert name in all_set, f"{name} not in __all__"


class TestExportsAreCallable:
    """Functions listed in exports should be callable."""

    FUNCTION_EXPORTS = [
        "configure_logging",
        "get_logger",
        "get_tracer",
        "create_span",
        "trace_handler",
        "trace_async_handler",
        "trace_agent_call",
        "trace_debate",
        "trace_debate_phase",
        "trace_consensus_check",
        "trace_memory_operation",
        "add_span_attributes",
        "record_exception",
        "shutdown_tracing",
        "start_metrics_server",
        "record_request",
        "record_agent_call",
        "set_consensus_rate",
        "record_memory_operation",
        "measure_latency",
        "measure_async_latency",
        "record_debate_completion",
        "record_phase_duration",
        "record_agent_participation",
        "record_cache_hit",
        "record_cache_miss",
        "get_tracing_config",
        "get_metrics_config",
        "is_tracing_enabled",
        "is_metrics_enabled",
        "set_correlation_id",
        "get_correlation_id",
    ]

    @pytest.mark.parametrize("name", FUNCTION_EXPORTS)
    def test_function_is_callable(self, name: str):
        from aragora import telemetry

        obj = getattr(telemetry, name)
        assert callable(obj), f"{name} should be callable"


# ---------------------------------------------------------------------------
# Basic instantiation (no external deps required)
# ---------------------------------------------------------------------------


class TestLogConfig:
    """LogConfig dataclass can be created with defaults."""

    def test_default_instantiation(self):
        from aragora.telemetry import LogConfig

        config = LogConfig()
        assert config.environment == "development"
        assert config.level == "INFO"
        assert config.format == "human"

    def test_custom_values(self):
        from aragora.telemetry import LogConfig

        config = LogConfig(environment="production", level="WARNING", format="json")
        assert config.environment == "production"
        assert config.level == "WARNING"
        assert config.format == "json"


class TestTracingConfig:
    """TracingConfig dataclass can be created with defaults."""

    def test_default_instantiation(self):
        from aragora.telemetry import TracingConfig

        config = TracingConfig()
        assert config.enabled is False
        assert config.service_name == "aragora"
        assert 0.0 <= config.sample_rate <= 1.0

    def test_enable_tracing(self):
        from aragora.telemetry import TracingConfig

        config = TracingConfig(enabled=True, sample_rate=0.5)
        assert config.enabled is True
        assert config.sample_rate == 0.5


class TestMetricsConfig:
    """MetricsConfig dataclass can be created with defaults."""

    def test_default_instantiation(self):
        from aragora.telemetry import MetricsConfig

        config = MetricsConfig()
        assert config.enabled is True
        assert config.port == 9090
        assert config.path == "/metrics"

    def test_custom_port(self):
        from aragora.telemetry import MetricsConfig

        config = MetricsConfig(port=8888, enabled=False)
        assert config.port == 8888
        assert config.enabled is False


class TestStructuredLogger:
    """StructuredLogger can be created with a stdlib logger and LogConfig."""

    def test_instantiation(self):
        from aragora.telemetry import LogConfig, StructuredLogger

        stdlib_logger = logging.getLogger("test.telemetry")
        config = LogConfig()
        logger = StructuredLogger(stdlib_logger, config)
        assert logger is not None

    def test_get_logger_returns_structured_logger(self):
        from aragora.telemetry import StructuredLogger, get_logger

        logger = get_logger("test.telemetry.exports")
        assert isinstance(logger, StructuredLogger)


class TestConfigurableLogging:
    """configure_logging returns a LogConfig and does not raise."""

    def test_configure_defaults(self):
        from aragora.telemetry import LogConfig, configure_logging

        config = configure_logging()
        assert isinstance(config, LogConfig)

    def test_configure_with_params(self):
        from aragora.telemetry import configure_logging

        config = configure_logging(environment="test", level="DEBUG", format="json")
        assert config.environment == "test"
        assert config.level == "DEBUG"


class TestCorrelation:
    """Correlation ID helpers work without external deps."""

    def test_set_and_get_correlation_id(self):
        from aragora.telemetry import get_correlation_id, set_correlation_id

        set_correlation_id("test-corr-123")
        assert get_correlation_id() == "test-corr-123"

    def test_correlation_context(self):
        from aragora.telemetry import correlation_context, get_correlation_id

        with correlation_context("ctx-456"):
            assert get_correlation_id() == "ctx-456"


class TestTracingHelpers:
    """Tracing helper functions are accessible and return sensible values."""

    def test_get_tracing_config(self):
        from aragora.telemetry import TracingConfig, get_tracing_config

        config = get_tracing_config()
        assert isinstance(config, TracingConfig)

    def test_get_metrics_config(self):
        from aragora.telemetry import MetricsConfig, get_metrics_config

        config = get_metrics_config()
        assert isinstance(config, MetricsConfig)

    def test_is_tracing_enabled_returns_bool(self):
        from aragora.telemetry import is_tracing_enabled

        result = is_tracing_enabled()
        assert isinstance(result, bool)

    def test_is_metrics_enabled_returns_bool(self):
        from aragora.telemetry import is_metrics_enabled

        result = is_metrics_enabled()
        assert isinstance(result, bool)
