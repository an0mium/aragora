"""
Tests for OTLP collector connectivity, protocol selection, and health checks.

Tests verify:
- OTelConfig protocol and headers fields
- Protocol/headers parsing from environment variables
- _create_otlp_exporter factory with gRPC/HTTP fallback
- check_otlp_health connectivity probe
- get_otel_status summary
- Server startup check_otlp_connectivity integration
"""

import os
from unittest.mock import patch

import pytest

from aragora.observability.otel import (
    OTelConfig,
    OTLPHealthStatus,
    _create_otlp_exporter,
    check_otlp_health,
    get_otel_status,
    reset_otel,
)


# =============================================================================
# Protocol and headers configuration tests
# =============================================================================


class TestProtocolConfig:
    """Tests for protocol and headers configuration on OTelConfig."""

    def test_default_protocol_is_grpc(self):
        """Test default protocol is grpc."""
        config = OTelConfig()
        assert config.protocol == "grpc"

    def test_http_protocol(self):
        """Test http/protobuf protocol is accepted."""
        config = OTelConfig(protocol="http/protobuf")
        assert config.protocol == "http/protobuf"

    def test_invalid_protocol_raises(self):
        """Test invalid protocol raises ValueError."""
        with pytest.raises(ValueError, match="protocol must be"):
            OTelConfig(protocol="invalid")

    def test_default_headers_empty(self):
        """Test default headers are empty dict."""
        config = OTelConfig()
        assert config.headers == {}

    def test_custom_headers(self):
        """Test custom headers are stored."""
        config = OTelConfig(headers={"Authorization": "Bearer token"})
        assert config.headers == {"Authorization": "Bearer token"}

    def test_protocol_grpc_string(self):
        """Test grpc string is valid."""
        config = OTelConfig(protocol="grpc")
        assert config.protocol == "grpc"

    def test_multiple_headers(self):
        """Test multiple headers are stored."""
        headers = {"X-Api-Key": "key123", "X-Custom": "value"}
        config = OTelConfig(headers=headers)
        assert config.headers == headers


# =============================================================================
# Protocol and headers environment variable parsing tests
# =============================================================================


class TestProtocolFromEnv:
    """Tests for protocol and headers parsing from environment variables."""

    def setup_method(self):
        """Clear environment variables before each test."""
        self._env_vars = [
            "OTEL_EXPORTER_OTLP_PROTOCOL",
            "ARAGORA_OTLP_PROTOCOL",
            "OTEL_EXPORTER_OTLP_HEADERS",
            "ARAGORA_OTLP_HEADERS",
            "OTEL_ENABLED",
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "ARAGORA_OTLP_EXPORTER",
        ]
        self._saved = {}
        for var in self._env_vars:
            self._saved[var] = os.environ.pop(var, None)

    def teardown_method(self):
        """Restore environment variables after each test."""
        for var in self._env_vars:
            os.environ.pop(var, None)
            if self._saved.get(var) is not None:
                os.environ[var] = self._saved[var]

    def test_default_protocol_from_env(self):
        """Test default protocol from env is grpc."""
        config = OTelConfig.from_env()
        assert config.protocol == "grpc"

    def test_otel_protocol_env(self):
        """Test OTEL_EXPORTER_OTLP_PROTOCOL env var is parsed."""
        os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"
        config = OTelConfig.from_env()
        assert config.protocol == "http/protobuf"

    def test_aragora_protocol_env_fallback(self):
        """Test ARAGORA_OTLP_PROTOCOL is used when OTEL var not set."""
        os.environ["ARAGORA_OTLP_PROTOCOL"] = "http/protobuf"
        config = OTelConfig.from_env()
        assert config.protocol == "http/protobuf"

    def test_otel_protocol_takes_precedence(self):
        """Test OTEL_ var takes precedence over ARAGORA_ var."""
        os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "grpc"
        os.environ["ARAGORA_OTLP_PROTOCOL"] = "http/protobuf"
        config = OTelConfig.from_env()
        assert config.protocol == "grpc"

    def test_invalid_protocol_falls_back_to_grpc(self):
        """Test invalid protocol falls back to grpc with warning."""
        os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "invalid_proto"
        config = OTelConfig.from_env()
        assert config.protocol == "grpc"

    def test_headers_json_format(self):
        """Test headers parsed from JSON format."""
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = '{"Authorization": "Bearer abc"}'
        config = OTelConfig.from_env()
        assert config.headers == {"Authorization": "Bearer abc"}

    def test_headers_key_value_format(self):
        """Test headers parsed from comma-separated key=value format."""
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = "api-key=abc123,x-custom=val"
        config = OTelConfig.from_env()
        assert config.headers == {"api-key": "abc123", "x-custom": "val"}

    def test_aragora_headers_fallback(self):
        """Test ARAGORA_OTLP_HEADERS env var as fallback."""
        os.environ["ARAGORA_OTLP_HEADERS"] = '{"X-Token": "t123"}'
        config = OTelConfig.from_env()
        assert config.headers == {"X-Token": "t123"}

    def test_empty_headers_default(self):
        """Test empty headers when no env vars set."""
        config = OTelConfig.from_env()
        assert config.headers == {}

    def test_malformed_json_headers_fallback_key_value(self):
        """Test malformed JSON headers tries key=value parsing."""
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = "not-json-but=parseable"
        config = OTelConfig.from_env()
        assert config.headers == {"not-json-but": "parseable"}


# =============================================================================
# Exporter factory tests
# =============================================================================


class TestCreateOtlpExporter:
    """Tests for _create_otlp_exporter factory function."""

    def test_grpc_exporter_creation(self):
        """Test gRPC exporter is created when available."""
        config = OTelConfig(
            endpoint="http://localhost:4317",
            protocol="grpc",
            insecure=True,
        )
        exporter = _create_otlp_exporter(config)
        # If any OTLP exporter package is installed, we get a result
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            assert exporter is not None
        except ImportError:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )
                # Fell back to HTTP
                assert exporter is not None
            except ImportError:
                assert exporter is None

    def test_http_exporter_creation(self):
        """Test HTTP exporter is created when protocol=http/protobuf."""
        config = OTelConfig(
            endpoint="http://localhost:4318",
            protocol="http/protobuf",
        )
        exporter = _create_otlp_exporter(config)
        # Either HTTP or gRPC fallback should work
        has_any = False
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            has_any = True
        except ImportError:
            pass
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            has_any = True
        except ImportError:
            pass

        if has_any:
            assert exporter is not None
        else:
            assert exporter is None

    def test_grpc_with_headers(self):
        """Test gRPC exporter passes headers as list of tuples."""
        config = OTelConfig(
            endpoint="http://localhost:4317",
            protocol="grpc",
            insecure=True,
            headers={"Authorization": "Bearer token"},
        )
        exporter = _create_otlp_exporter(config)
        # Should not raise even with headers
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            assert exporter is not None
        except ImportError:
            pass

    def test_factory_returns_none_when_no_exporters(self):
        """Test factory returns None when no exporter packages installed."""
        config = OTelConfig(protocol="grpc", insecure=True)

        # Mock both exporter imports to fail
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def mock_import(name, *args, **kwargs):
            if "opentelemetry.exporter.otlp" in name:
                raise ImportError(f"Mocked: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = _create_otlp_exporter(config)
            assert result is None


# =============================================================================
# OTLP Health Status tests
# =============================================================================


class TestOTLPHealthStatus:
    """Tests for OTLPHealthStatus dataclass."""

    def test_healthy_status(self):
        """Test creating a healthy status."""
        status = OTLPHealthStatus(
            healthy=True,
            endpoint="http://localhost:4317",
            protocol="grpc",
            latency_ms=5.2,
            initialized=True,
        )
        assert status.healthy is True
        assert status.endpoint == "http://localhost:4317"
        assert status.protocol == "grpc"
        assert status.latency_ms == 5.2
        assert status.error is None
        assert status.initialized is True

    def test_unhealthy_status(self):
        """Test creating an unhealthy status."""
        status = OTLPHealthStatus(
            healthy=False,
            endpoint="http://collector:4317",
            protocol="grpc",
            latency_ms=3000.0,
            error="Connection timed out after 3000ms",
            initialized=False,
        )
        assert status.healthy is False
        assert status.error == "Connection timed out after 3000ms"
        assert status.initialized is False

    def test_defaults(self):
        """Test default optional fields."""
        status = OTLPHealthStatus(
            healthy=True,
            endpoint="http://localhost:4317",
            protocol="grpc",
        )
        assert status.latency_ms is None
        assert status.error is None
        assert status.initialized is False


# =============================================================================
# Health check tests
# =============================================================================


class TestCheckOtlpHealth:
    """Tests for check_otlp_health function."""

    def setup_method(self):
        reset_otel()

    def teardown_method(self):
        reset_otel()

    def test_connection_refused(self):
        """Test health check when no collector is running."""
        status = check_otlp_health(
            endpoint="http://localhost:19999",
            timeout_ms=500,
        )
        assert status.healthy is False
        assert status.latency_ms is not None
        assert status.latency_ms >= 0
        assert status.error is not None

    def test_timeout_unreachable_host(self):
        """Test health check timeout with non-routable address."""
        status = check_otlp_health(
            endpoint="http://192.0.2.1:4317",  # TEST-NET, non-routable
            timeout_ms=500,
        )
        assert status.healthy is False
        assert status.error is not None

    def test_default_grpc_port(self):
        """Test health check defaults to port 4317 for gRPC."""
        status = check_otlp_health(
            endpoint="http://localhost",
            timeout_ms=200,
        )
        assert status.endpoint == "http://localhost"
        assert status.protocol == "grpc"

    def test_initialized_flag_reflects_state(self):
        """Test health check reflects OTel initialization state."""
        status = check_otlp_health(
            endpoint="http://localhost:19999",
            timeout_ms=200,
        )
        assert status.initialized is False

    def test_explicit_endpoint_used(self):
        """Test explicit endpoint overrides config."""
        status = check_otlp_health(
            endpoint="http://127.0.0.1:19999",
            timeout_ms=200,
        )
        assert status.endpoint == "http://127.0.0.1:19999"

    def test_returns_positive_latency(self):
        """Test latency_ms is always positive."""
        status = check_otlp_health(
            endpoint="http://localhost:19999",
            timeout_ms=200,
        )
        assert status.latency_ms is not None
        assert status.latency_ms >= 0


# =============================================================================
# OTel status summary tests
# =============================================================================


class TestGetOtelStatus:
    """Tests for get_otel_status function."""

    def setup_method(self):
        reset_otel()

    def teardown_method(self):
        reset_otel()

    def test_not_initialized(self):
        """Test status reports not initialized."""
        status = get_otel_status()
        assert status["initialized"] is False
        assert "reason" in status

    def test_returns_dict(self):
        """Test status always returns a dict."""
        status = get_otel_status()
        assert isinstance(status, dict)

    def test_not_initialized_has_reason(self):
        """Test reason explains how to enable."""
        status = get_otel_status()
        assert "OTEL_EXPORTER_OTLP_ENDPOINT" in status.get("reason", "")


# =============================================================================
# Server startup integration tests
# =============================================================================


class TestStartupOtlpConnectivity:
    """Tests for the server startup OTLP connectivity check."""

    def setup_method(self):
        self._saved = {}
        for var in ("OTEL_ENABLED", "OTEL_EXPORTER_OTLP_ENDPOINT", "ARAGORA_OTLP_EXPORTER"):
            self._saved[var] = os.environ.pop(var, None)

    def teardown_method(self):
        for var, val in self._saved.items():
            os.environ.pop(var, None)
            if val is not None:
                os.environ[var] = val

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self):
        """Test connectivity check skips when tracing is not enabled."""
        from aragora.server.startup.observability import check_otlp_connectivity

        result = await check_otlp_connectivity()
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_unreachable(self):
        """Test connectivity check returns False when collector is unreachable."""
        from aragora.server.startup.observability import check_otlp_connectivity

        os.environ["OTEL_ENABLED"] = "true"
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:19999"

        result = await check_otlp_connectivity()
        assert result is False


# =============================================================================
# Package export tests
# =============================================================================


class TestPackageExports:
    """Test new symbols are importable from the observability package."""

    def test_otlp_health_status_importable(self):
        """Test OTLPHealthStatus is importable from observability package."""
        from aragora.observability import OTLPHealthStatus as HS

        assert HS is OTLPHealthStatus

    def test_check_otlp_health_importable(self):
        """Test check_otlp_health is importable from observability package."""
        from aragora.observability import check_otlp_health as fn

        assert fn is check_otlp_health

    def test_get_otel_status_importable(self):
        """Test get_otel_status is importable from observability package."""
        from aragora.observability import get_otel_status as fn

        assert fn is get_otel_status
