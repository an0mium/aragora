"""
Tests for OpenTelemetry OTLP export configuration.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from aragora.observability.otlp_export import (
    DEFAULT_ENDPOINTS,
    OTLPConfig,
    OTLPExporterType,
    configure_otlp_exporter,
    get_otlp_config,
    get_tracer_provider,
    reset_otlp_config,
    set_otlp_config,
    shutdown_otlp,
)


class TestOTLPExporterType:
    """Tests for OTLPExporterType enum."""

    def test_exporter_types_exist(self):
        """Test all expected exporter types exist."""
        assert OTLPExporterType.NONE == "none"
        assert OTLPExporterType.JAEGER == "jaeger"
        assert OTLPExporterType.ZIPKIN == "zipkin"
        assert OTLPExporterType.OTLP_GRPC == "otlp_grpc"
        assert OTLPExporterType.OTLP_HTTP == "otlp_http"
        assert OTLPExporterType.DATADOG == "datadog"

    def test_exporter_type_from_string(self):
        """Test creating exporter type from string."""
        assert OTLPExporterType("none") == OTLPExporterType.NONE
        assert OTLPExporterType("jaeger") == OTLPExporterType.JAEGER
        assert OTLPExporterType("zipkin") == OTLPExporterType.ZIPKIN
        assert OTLPExporterType("otlp_grpc") == OTLPExporterType.OTLP_GRPC
        assert OTLPExporterType("otlp_http") == OTLPExporterType.OTLP_HTTP
        assert OTLPExporterType("datadog") == OTLPExporterType.DATADOG

    def test_invalid_exporter_type_raises(self):
        """Test invalid exporter type raises ValueError."""
        with pytest.raises(ValueError):
            OTLPExporterType("invalid")


class TestDefaultEndpoints:
    """Tests for default endpoint mappings."""

    def test_default_endpoints_defined(self):
        """Test default endpoints are defined for all exporters except NONE."""
        assert OTLPExporterType.JAEGER in DEFAULT_ENDPOINTS
        assert OTLPExporterType.ZIPKIN in DEFAULT_ENDPOINTS
        assert OTLPExporterType.OTLP_GRPC in DEFAULT_ENDPOINTS
        assert OTLPExporterType.OTLP_HTTP in DEFAULT_ENDPOINTS
        assert OTLPExporterType.DATADOG in DEFAULT_ENDPOINTS

    def test_default_endpoint_values(self):
        """Test default endpoint values are correct."""
        assert DEFAULT_ENDPOINTS[OTLPExporterType.JAEGER] == "localhost"
        assert DEFAULT_ENDPOINTS[OTLPExporterType.ZIPKIN] == "http://localhost:9411/api/v2/spans"
        assert DEFAULT_ENDPOINTS[OTLPExporterType.OTLP_GRPC] == "http://localhost:4317"
        assert DEFAULT_ENDPOINTS[OTLPExporterType.OTLP_HTTP] == "http://localhost:4318/v1/traces"
        assert DEFAULT_ENDPOINTS[OTLPExporterType.DATADOG] == "http://localhost:4317"


class TestOTLPConfig:
    """Tests for OTLPConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OTLPConfig()
        assert config.exporter_type == OTLPExporterType.NONE
        assert config.endpoint is None
        assert config.service_name == "aragora"
        assert config.service_version == "1.0.0"
        assert config.environment == "development"
        assert config.sample_rate == 1.0
        assert config.headers == {}
        assert config.batch_size == 512
        assert config.export_timeout_ms == 30000
        assert config.insecure is False
        assert config.datadog_api_key is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = OTLPConfig(
            exporter_type=OTLPExporterType.JAEGER,
            endpoint="jaeger.example.com",
            service_name="my-service",
            service_version="2.0.0",
            environment="production",
            sample_rate=0.5,
            headers={"Authorization": "Bearer token"},
            batch_size=1024,
            export_timeout_ms=60000,
            insecure=True,
            datadog_api_key="dd-key",
        )
        assert config.exporter_type == OTLPExporterType.JAEGER
        assert config.endpoint == "jaeger.example.com"
        assert config.service_name == "my-service"
        assert config.service_version == "2.0.0"
        assert config.environment == "production"
        assert config.sample_rate == 0.5
        assert config.headers == {"Authorization": "Bearer token"}
        assert config.batch_size == 1024
        assert config.export_timeout_ms == 60000
        assert config.insecure is True
        assert config.datadog_api_key == "dd-key"

    def test_sample_rate_validation(self):
        """Test sample rate validation."""
        # Valid rates
        OTLPConfig(sample_rate=0.0)
        OTLPConfig(sample_rate=0.5)
        OTLPConfig(sample_rate=1.0)

        # Invalid rates
        with pytest.raises(ValueError, match="sample_rate must be between"):
            OTLPConfig(sample_rate=-0.1)

        with pytest.raises(ValueError, match="sample_rate must be between"):
            OTLPConfig(sample_rate=1.1)

    def test_batch_size_validation(self):
        """Test batch size validation."""
        # Valid batch size
        OTLPConfig(batch_size=1)
        OTLPConfig(batch_size=1000)

        # Invalid batch size
        with pytest.raises(ValueError, match="batch_size must be positive"):
            OTLPConfig(batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            OTLPConfig(batch_size=-1)

    def test_export_timeout_validation(self):
        """Test export timeout validation."""
        # Valid timeout
        OTLPConfig(export_timeout_ms=1)
        OTLPConfig(export_timeout_ms=60000)

        # Invalid timeout
        with pytest.raises(ValueError, match="export_timeout_ms must be positive"):
            OTLPConfig(export_timeout_ms=0)

        with pytest.raises(ValueError, match="export_timeout_ms must be positive"):
            OTLPConfig(export_timeout_ms=-1)

    def test_get_effective_endpoint_none(self):
        """Test get_effective_endpoint returns None for NONE type."""
        config = OTLPConfig(exporter_type=OTLPExporterType.NONE)
        assert config.get_effective_endpoint() is None

    def test_get_effective_endpoint_default(self):
        """Test get_effective_endpoint returns default when no endpoint specified."""
        config = OTLPConfig(exporter_type=OTLPExporterType.JAEGER)
        assert config.get_effective_endpoint() == "localhost"

        config = OTLPConfig(exporter_type=OTLPExporterType.ZIPKIN)
        assert config.get_effective_endpoint() == "http://localhost:9411/api/v2/spans"

    def test_get_effective_endpoint_custom(self):
        """Test get_effective_endpoint returns custom endpoint when specified."""
        config = OTLPConfig(
            exporter_type=OTLPExporterType.JAEGER,
            endpoint="custom.example.com",
        )
        assert config.get_effective_endpoint() == "custom.example.com"


class TestOTLPConfigFromEnv:
    """Tests for OTLPConfig.from_env()."""

    def setup_method(self):
        """Clear environment before each test."""
        self._clear_env()
        reset_otlp_config()

    def teardown_method(self):
        """Clear environment after each test."""
        self._clear_env()
        reset_otlp_config()

    def _clear_env(self):
        """Clear OTLP-related environment variables."""
        vars_to_clear = [
            "ARAGORA_OTLP_EXPORTER",
            "ARAGORA_OTLP_ENDPOINT",
            "ARAGORA_SERVICE_NAME",
            "ARAGORA_SERVICE_VERSION",
            "ARAGORA_ENVIRONMENT",
            "ARAGORA_TRACE_SAMPLE_RATE",
            "ARAGORA_OTLP_HEADERS",
            "ARAGORA_OTLP_BATCH_SIZE",
            "ARAGORA_OTLP_EXPORT_TIMEOUT_MS",
            "ARAGORA_OTLP_INSECURE",
            "DATADOG_API_KEY",
        ]
        for var in vars_to_clear:
            os.environ.pop(var, None)

    def test_from_env_defaults(self):
        """Test from_env with no environment variables set."""
        config = OTLPConfig.from_env()
        assert config.exporter_type == OTLPExporterType.NONE
        assert config.endpoint is None
        assert config.service_name == "aragora"
        assert config.environment == "development"
        assert config.sample_rate == 1.0

    def test_from_env_exporter_type(self):
        """Test from_env parses exporter type."""
        os.environ["ARAGORA_OTLP_EXPORTER"] = "jaeger"
        config = OTLPConfig.from_env()
        assert config.exporter_type == OTLPExporterType.JAEGER

        os.environ["ARAGORA_OTLP_EXPORTER"] = "ZIPKIN"
        config = OTLPConfig.from_env()
        assert config.exporter_type == OTLPExporterType.ZIPKIN

    def test_from_env_invalid_exporter_defaults_to_none(self):
        """Test from_env defaults to NONE for invalid exporter type."""
        os.environ["ARAGORA_OTLP_EXPORTER"] = "invalid_type"
        config = OTLPConfig.from_env()
        assert config.exporter_type == OTLPExporterType.NONE

    def test_from_env_endpoint(self):
        """Test from_env parses endpoint."""
        os.environ["ARAGORA_OTLP_ENDPOINT"] = "http://custom.example.com:4317"
        config = OTLPConfig.from_env()
        assert config.endpoint == "http://custom.example.com:4317"

    def test_from_env_service_name(self):
        """Test from_env parses service name."""
        os.environ["ARAGORA_SERVICE_NAME"] = "my-service"
        config = OTLPConfig.from_env()
        assert config.service_name == "my-service"

    def test_from_env_sample_rate(self):
        """Test from_env parses sample rate."""
        os.environ["ARAGORA_TRACE_SAMPLE_RATE"] = "0.5"
        config = OTLPConfig.from_env()
        assert config.sample_rate == 0.5

    def test_from_env_headers_json(self):
        """Test from_env parses JSON headers."""
        headers = {"Authorization": "Bearer token", "X-Custom": "value"}
        os.environ["ARAGORA_OTLP_HEADERS"] = json.dumps(headers)
        config = OTLPConfig.from_env()
        assert config.headers == headers

    def test_from_env_headers_invalid_json(self):
        """Test from_env handles invalid JSON headers gracefully."""
        os.environ["ARAGORA_OTLP_HEADERS"] = "not valid json"
        config = OTLPConfig.from_env()
        assert config.headers == {}

    def test_from_env_batch_size(self):
        """Test from_env parses batch size."""
        os.environ["ARAGORA_OTLP_BATCH_SIZE"] = "1024"
        config = OTLPConfig.from_env()
        assert config.batch_size == 1024

    def test_from_env_insecure(self):
        """Test from_env parses insecure flag."""
        os.environ["ARAGORA_OTLP_INSECURE"] = "true"
        config = OTLPConfig.from_env()
        assert config.insecure is True

        os.environ["ARAGORA_OTLP_INSECURE"] = "1"
        config = OTLPConfig.from_env()
        assert config.insecure is True

        os.environ["ARAGORA_OTLP_INSECURE"] = "false"
        config = OTLPConfig.from_env()
        assert config.insecure is False

    def test_from_env_datadog_api_key(self):
        """Test from_env parses Datadog API key."""
        os.environ["DATADOG_API_KEY"] = "dd-api-key-123"
        config = OTLPConfig.from_env()
        assert config.datadog_api_key == "dd-api-key-123"


class TestConfigSingletons:
    """Tests for configuration singleton functions."""

    def setup_method(self):
        """Reset config before each test."""
        reset_otlp_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_otlp_config()

    def test_get_otlp_config_returns_config(self):
        """Test get_otlp_config returns a config."""
        config = get_otlp_config()
        assert isinstance(config, OTLPConfig)

    def test_set_otlp_config(self):
        """Test set_otlp_config sets custom config."""
        custom_config = OTLPConfig(
            exporter_type=OTLPExporterType.JAEGER,
            service_name="custom-service",
        )
        set_otlp_config(custom_config)
        config = get_otlp_config()
        assert config.exporter_type == OTLPExporterType.JAEGER
        assert config.service_name == "custom-service"

    def test_reset_otlp_config(self):
        """Test reset_otlp_config resets to defaults."""
        custom_config = OTLPConfig(service_name="custom-service")
        set_otlp_config(custom_config)
        reset_otlp_config()
        config = get_otlp_config()
        assert config.service_name == "aragora"


class TestConfigureOTLPExporter:
    """Tests for configure_otlp_exporter function."""

    def setup_method(self):
        """Reset config before each test."""
        reset_otlp_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_otlp_config()
        shutdown_otlp()

    def test_configure_returns_none_for_none_exporter(self):
        """Test configure returns None when exporter type is NONE."""
        config = OTLPConfig(exporter_type=OTLPExporterType.NONE)
        result = configure_otlp_exporter(config)
        assert result is None

    def test_configure_with_jaeger_missing_package(self):
        """Test configure handles missing Jaeger package."""
        config = OTLPConfig(exporter_type=OTLPExporterType.JAEGER)
        with patch.dict("sys.modules", {"opentelemetry.exporter.jaeger.thrift": None}):
            # This will try to import and fail gracefully
            result = configure_otlp_exporter(config)
            # May or may not return provider depending on if package is installed

    def test_configure_with_valid_config(self):
        """Test configure with valid OTLP gRPC config."""
        config = OTLPConfig(
            exporter_type=OTLPExporterType.OTLP_GRPC,
            endpoint="http://localhost:4317",
            service_name="test-service",
        )

        # Mock the OpenTelemetry imports to test the flow
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider

            result = configure_otlp_exporter(config)
            # Result depends on whether packages are installed
            if result is not None:
                assert isinstance(result, TracerProvider)
        except ImportError:
            # If OpenTelemetry is not installed, configure should return None gracefully
            result = configure_otlp_exporter(config)
            assert result is None

    def test_get_tracer_provider(self):
        """Test get_tracer_provider returns configured provider."""
        # Initially should be None
        reset_otlp_config()
        provider = get_tracer_provider()
        assert provider is None

    def test_shutdown_otlp(self):
        """Test shutdown_otlp handles None provider."""
        reset_otlp_config()
        # Should not raise
        shutdown_otlp()


class TestExporterFactories:
    """Tests for individual exporter factory functions."""

    def test_jaeger_exporter_creation(self):
        """Test Jaeger exporter factory handles missing package gracefully."""
        from aragora.observability.otlp_export import _get_jaeger_exporter

        config = OTLPConfig(
            exporter_type=OTLPExporterType.JAEGER,
            endpoint="jaeger.example.com",
        )

        # The function should not raise, it returns None if package not installed
        exporter = _get_jaeger_exporter(config)
        # Exporter may be None if package not installed - that's expected
        # If package IS installed, we verify we get an exporter
        try:
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter

            # Package is available, so exporter should not be None
            assert exporter is not None
        except ImportError:
            # Package not installed, None is expected
            assert exporter is None

    def test_zipkin_exporter_creation(self):
        """Test Zipkin exporter factory handles missing package gracefully."""
        from aragora.observability.otlp_export import _get_zipkin_exporter

        config = OTLPConfig(
            exporter_type=OTLPExporterType.ZIPKIN,
            endpoint="http://zipkin.example.com:9411/api/v2/spans",
        )

        exporter = _get_zipkin_exporter(config)
        try:
            from opentelemetry.exporter.zipkin.json import ZipkinExporter

            assert exporter is not None
        except ImportError:
            assert exporter is None

    def test_otlp_grpc_exporter_creation(self):
        """Test OTLP gRPC exporter factory handles missing package gracefully."""
        from aragora.observability.otlp_export import _get_otlp_grpc_exporter

        config = OTLPConfig(
            exporter_type=OTLPExporterType.OTLP_GRPC,
            endpoint="http://localhost:4317",
            headers={"Authorization": "Bearer token"},
            insecure=True,
            export_timeout_ms=5000,
        )

        exporter = _get_otlp_grpc_exporter(config)
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            assert exporter is not None
        except ImportError:
            assert exporter is None

    def test_otlp_http_exporter_creation(self):
        """Test OTLP HTTP exporter factory handles missing package gracefully."""
        from aragora.observability.otlp_export import _get_otlp_http_exporter

        config = OTLPConfig(
            exporter_type=OTLPExporterType.OTLP_HTTP,
            endpoint="http://localhost:4318/v1/traces",
        )

        exporter = _get_otlp_http_exporter(config)
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            assert exporter is not None
        except ImportError:
            assert exporter is None

    def test_datadog_exporter_creation(self):
        """Test Datadog exporter factory handles missing package gracefully."""
        from aragora.observability.otlp_export import _get_datadog_exporter

        config = OTLPConfig(
            exporter_type=OTLPExporterType.DATADOG,
            endpoint="http://localhost:4317",
            datadog_api_key="dd-api-key-123",
        )

        exporter = _get_datadog_exporter(config)
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            assert exporter is not None
        except ImportError:
            assert exporter is None


class TestExporterSelection:
    """Tests for exporter selection based on config."""

    def test_get_exporter_none(self):
        """Test _get_exporter returns None for NONE type."""
        from aragora.observability.otlp_export import _get_exporter

        config = OTLPConfig(exporter_type=OTLPExporterType.NONE)
        assert _get_exporter(config) is None

    def test_get_exporter_routes_correctly(self):
        """Test _get_exporter routes to correct factory."""
        from aragora.observability.otlp_export import _get_exporter

        # Test each exporter type is routed correctly
        for exporter_type in [
            OTLPExporterType.JAEGER,
            OTLPExporterType.ZIPKIN,
            OTLPExporterType.OTLP_GRPC,
            OTLPExporterType.OTLP_HTTP,
            OTLPExporterType.DATADOG,
        ]:
            config = OTLPConfig(exporter_type=exporter_type)
            # Should not raise, may return None if package not installed
            _get_exporter(config)


class TestIntegration:
    """Integration tests for OTLP export."""

    def setup_method(self):
        """Reset config before each test."""
        reset_otlp_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_otlp_config()
        shutdown_otlp()

    def test_full_configuration_flow(self):
        """Test complete configuration flow."""
        # Create config
        config = OTLPConfig(
            exporter_type=OTLPExporterType.OTLP_GRPC,
            endpoint="http://localhost:4317",
            service_name="integration-test",
            environment="test",
            sample_rate=0.5,
        )

        # Set config
        set_otlp_config(config)

        # Verify config was set
        retrieved = get_otlp_config()
        assert retrieved.service_name == "integration-test"
        assert retrieved.environment == "test"
        assert retrieved.sample_rate == 0.5

        # Configure exporter
        provider = configure_otlp_exporter(config)

        # Shutdown cleanly
        shutdown_otlp()

    def test_is_otlp_enabled_config_check(self):
        """Test is_otlp_enabled returns correct values."""
        from aragora.observability.config import is_otlp_enabled

        # Default should be disabled
        reset_otlp_config()
        assert is_otlp_enabled() is False

        # Enable with JAEGER
        set_otlp_config(OTLPConfig(exporter_type=OTLPExporterType.JAEGER))
        assert is_otlp_enabled() is True

        # Disable with NONE
        set_otlp_config(OTLPConfig(exporter_type=OTLPExporterType.NONE))
        assert is_otlp_enabled() is False
