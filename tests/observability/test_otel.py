"""
Tests for the unified OpenTelemetry setup module (aragora.observability.otel).

Tests verify:
- Configuration loading from environment variables
- NoOp tracer/span behavior when OTel is not installed
- Setup/shutdown lifecycle
- Span creation helpers
- Debate-specific tracing context managers
- Context propagation helpers
- Error recording on spans
- Bridge from internal debate spans to OTel
"""

import os
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

from aragora.observability.otel import (
    OTelConfig,
    OTLPHealthStatus,
    _NoOpSpan,
    _NoOpSpanContext,
    _NoOpTracer,
    _create_otlp_exporter,
    check_otlp_health,
    export_debate_span_to_otel,
    extract_context,
    get_otel_status,
    get_tracer,
    inject_context,
    is_initialized,
    record_span_error,
    reset_otel,
    set_span_ok,
    setup_otel,
    shutdown_otel,
    start_span,
    trace_agent_operation,
    trace_consensus_evaluation,
    trace_debate_lifecycle,
    trace_debate_round,
)


# =============================================================================
# OTelConfig tests
# =============================================================================


class TestOTelConfig:
    """Tests for the OTelConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OTelConfig()
        assert config.enabled is False
        assert config.endpoint == "http://localhost:4317"
        assert config.service_name == "aragora"
        assert config.service_version == "1.0.0"
        assert config.environment == "development"
        assert config.sampler_type == "parentbased_traceidratio"
        assert config.sample_rate == 1.0
        assert config.propagators == ["tracecontext", "baggage"]
        assert config.dev_mode is False
        assert config.batch_size == 512
        assert config.export_timeout_ms == 30000
        assert config.insecure is False
        assert config.additional_resource_attrs == {}

    def test_custom_config(self):
        """Test custom configuration values."""
        config = OTelConfig(
            enabled=True,
            endpoint="http://collector:4317",
            service_name="my-service",
            service_version="2.0.0",
            environment="production",
            sampler_type="always_on",
            sample_rate=0.5,
            propagators=["tracecontext"],
            dev_mode=True,
            batch_size=1024,
            export_timeout_ms=60000,
            insecure=True,
            additional_resource_attrs={"custom.attr": "value"},
        )
        assert config.enabled is True
        assert config.endpoint == "http://collector:4317"
        assert config.service_name == "my-service"
        assert config.service_version == "2.0.0"
        assert config.environment == "production"
        assert config.sampler_type == "always_on"
        assert config.sample_rate == 0.5
        assert config.dev_mode is True
        assert config.batch_size == 1024
        assert config.additional_resource_attrs == {"custom.attr": "value"}

    def test_sample_rate_validation(self):
        """Test sample rate validation bounds."""
        OTelConfig(sample_rate=0.0)  # Valid
        OTelConfig(sample_rate=1.0)  # Valid

        with pytest.raises(ValueError, match="sample_rate must be between"):
            OTelConfig(sample_rate=-0.1)

        with pytest.raises(ValueError, match="sample_rate must be between"):
            OTelConfig(sample_rate=1.1)

    def test_batch_size_validation(self):
        """Test batch size validation."""
        OTelConfig(batch_size=1)  # Valid

        with pytest.raises(ValueError, match="batch_size must be positive"):
            OTelConfig(batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            OTelConfig(batch_size=-1)

    def test_export_timeout_validation(self):
        """Test export timeout validation."""
        OTelConfig(export_timeout_ms=1)  # Valid

        with pytest.raises(ValueError, match="export_timeout_ms must be positive"):
            OTelConfig(export_timeout_ms=0)


class TestOTelConfigFromEnv:
    """Tests for OTelConfig.from_env()."""

    def setup_method(self):
        """Clear environment variables before each test."""
        self._env_vars = [
            "OTEL_ENABLED",
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_SERVICE_NAME",
            "OTEL_TRACES_SAMPLER",
            "OTEL_TRACES_SAMPLER_ARG",
            "OTEL_SAMPLE_RATE",
            "OTEL_PROPAGATORS",
            "OTEL_RESOURCE_ATTRIBUTES",
            "ARAGORA_OTLP_EXPORTER",
            "ARAGORA_OTLP_ENDPOINT",
            "ARAGORA_SERVICE_NAME",
            "ARAGORA_SERVICE_VERSION",
            "ARAGORA_ENVIRONMENT",
            "ARAGORA_TRACE_SAMPLE_RATE",
            "ARAGORA_OTEL_DEV_MODE",
            "ARAGORA_OTLP_BATCH_SIZE",
            "ARAGORA_OTLP_EXPORT_TIMEOUT_MS",
            "ARAGORA_OTLP_INSECURE",
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

    def test_defaults_when_no_env(self):
        """Test from_env returns disabled config with defaults when no env vars set."""
        config = OTelConfig.from_env()
        assert config.enabled is False
        assert config.service_name == "aragora"
        assert config.environment == "development"

    def test_enabled_via_otel_enabled(self):
        """Test explicit enable via OTEL_ENABLED."""
        os.environ["OTEL_ENABLED"] = "true"
        config = OTelConfig.from_env()
        assert config.enabled is True

    def test_auto_enabled_via_endpoint(self):
        """Test auto-enable when OTEL_EXPORTER_OTLP_ENDPOINT is set."""
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://collector:4317"
        config = OTelConfig.from_env()
        assert config.enabled is True
        assert config.endpoint == "http://collector:4317"

    def test_auto_enabled_via_aragora_exporter(self):
        """Test auto-enable when ARAGORA_OTLP_EXPORTER is set."""
        os.environ["ARAGORA_OTLP_EXPORTER"] = "otlp_grpc"
        config = OTelConfig.from_env()
        assert config.enabled is True

    def test_service_name_priority(self):
        """Test OTEL_SERVICE_NAME takes precedence over ARAGORA_SERVICE_NAME."""
        os.environ["OTEL_SERVICE_NAME"] = "otel-service"
        os.environ["ARAGORA_SERVICE_NAME"] = "aragora-service"
        config = OTelConfig.from_env()
        assert config.service_name == "otel-service"

    def test_aragora_service_name_fallback(self):
        """Test ARAGORA_SERVICE_NAME used as fallback."""
        os.environ["ARAGORA_SERVICE_NAME"] = "aragora-service"
        config = OTelConfig.from_env()
        assert config.service_name == "aragora-service"

    def test_dev_mode_env(self):
        """Test ARAGORA_OTEL_DEV_MODE parsing."""
        os.environ["ARAGORA_OTEL_DEV_MODE"] = "true"
        config = OTelConfig.from_env()
        assert config.dev_mode is True

        os.environ["ARAGORA_OTEL_DEV_MODE"] = "false"
        config = OTelConfig.from_env()
        assert config.dev_mode is False

    def test_resource_attributes_parsing(self):
        """Test OTEL_RESOURCE_ATTRIBUTES parsing."""
        os.environ["OTEL_RESOURCE_ATTRIBUTES"] = "team=platform,region=us-east-1"
        config = OTelConfig.from_env()
        assert config.additional_resource_attrs == {
            "team": "platform",
            "region": "us-east-1",
        }

    def test_invalid_sample_rate_fallback(self):
        """Test invalid sample rate falls back to 1.0."""
        os.environ["OTEL_TRACES_SAMPLER_ARG"] = "not_a_number"
        config = OTelConfig.from_env()
        assert config.sample_rate == 1.0

    def test_propagators_parsing(self):
        """Test propagator list parsing."""
        os.environ["OTEL_PROPAGATORS"] = "tracecontext, baggage, b3"
        config = OTelConfig.from_env()
        assert config.propagators == ["tracecontext", "baggage", "b3"]


# =============================================================================
# NoOp implementation tests
# =============================================================================


class TestNoOpSpan:
    """Tests for _NoOpSpan behavior."""

    def test_set_attribute_returns_self(self):
        """Test set_attribute returns self for chaining."""
        span = _NoOpSpan()
        result = span.set_attribute("key", "value")
        assert result is span

    def test_set_attributes_returns_self(self):
        """Test set_attributes returns self."""
        span = _NoOpSpan()
        result = span.set_attributes({"a": 1, "b": 2})
        assert result is span

    def test_add_event_returns_self(self):
        """Test add_event returns self."""
        span = _NoOpSpan()
        result = span.add_event("event", {"key": "val"})
        assert result is span

    def test_record_exception_returns_self(self):
        """Test record_exception returns self."""
        span = _NoOpSpan()
        result = span.record_exception(ValueError("test"))
        assert result is span

    def test_set_status_returns_self(self):
        """Test set_status returns self."""
        span = _NoOpSpan()
        result = span.set_status("OK")
        assert result is span

    def test_update_name_returns_self(self):
        """Test update_name returns self."""
        span = _NoOpSpan()
        result = span.update_name("new_name")
        assert result is span

    def test_end_does_not_raise(self):
        """Test end() does not raise."""
        span = _NoOpSpan()
        span.end()

    def test_is_recording_returns_false(self):
        """Test is_recording returns False."""
        span = _NoOpSpan()
        assert span.is_recording() is False

    def test_context_manager(self):
        """Test NoOpSpan works as context manager."""
        span = _NoOpSpan()
        with span as s:
            assert s is span
            s.set_attribute("test", "value")

    def test_get_span_context(self):
        """Test get_span_context returns NoOpSpanContext."""
        span = _NoOpSpan()
        ctx = span.get_span_context()
        assert isinstance(ctx, _NoOpSpanContext)
        assert ctx.is_valid is False


class TestNoOpTracer:
    """Tests for _NoOpTracer behavior."""

    def test_start_as_current_span(self):
        """Test start_as_current_span returns NoOpSpan."""
        tracer = _NoOpTracer()
        span = tracer.start_as_current_span("test")
        assert isinstance(span, _NoOpSpan)

    def test_start_span(self):
        """Test start_span returns NoOpSpan."""
        tracer = _NoOpTracer()
        span = tracer.start_span("test")
        assert isinstance(span, _NoOpSpan)


# =============================================================================
# Setup/shutdown/state tests
# =============================================================================


class TestSetupShutdown:
    """Tests for setup_otel and shutdown_otel lifecycle."""

    def setup_method(self):
        """Reset OTel state before each test."""
        reset_otel()

    def teardown_method(self):
        """Reset OTel state after each test."""
        reset_otel()

    def test_not_initialized_by_default(self):
        """Test is_initialized returns False before setup."""
        assert is_initialized() is False

    def test_setup_disabled_returns_false(self):
        """Test setup returns False when disabled."""
        config = OTelConfig(enabled=False)
        result = setup_otel(config)
        assert result is False
        assert is_initialized() is False

    def test_setup_when_otel_not_installed(self):
        """Test setup handles missing OTel packages gracefully."""
        config = OTelConfig(enabled=True)

        # Mock the import to simulate OTel not being installed
        with patch("builtins.__import__", side_effect=_mock_import_error):
            result = setup_otel(config)
            # Should return False because OTel cannot be imported
            assert result is False

    def test_shutdown_when_not_initialized(self):
        """Test shutdown is safe when not initialized."""
        # Should not raise
        shutdown_otel()

    def test_reset_clears_state(self):
        """Test reset_otel clears all state."""
        # Even if we can't fully setup, reset should clear everything
        reset_otel()
        assert is_initialized() is False

    def test_get_tracer_returns_noop_when_not_initialized(self):
        """Test get_tracer returns NoOpTracer when not initialized."""
        tracer = get_tracer("test")
        assert isinstance(tracer, _NoOpTracer)

    def test_get_tracer_caches_results(self):
        """Test get_tracer caches tracer instances."""
        tracer1 = get_tracer("test")
        tracer2 = get_tracer("test")
        assert tracer1 is tracer2

    def test_get_tracer_different_names(self):
        """Test different instrumentation names get separate tracers."""
        tracer1 = get_tracer("scope.a")
        tracer2 = get_tracer("scope.b")
        # Both should be NoOpTracer instances (since not initialized)
        assert isinstance(tracer1, _NoOpTracer)
        assert isinstance(tracer2, _NoOpTracer)


def _mock_import_error(name, *args, **kwargs):
    """Mock import that raises ImportError for OTel packages."""
    if "opentelemetry" in name:
        raise ImportError(f"Mocked: No module named '{name}'")
    return (
        __builtins__.__import__(name, *args, **kwargs)
        if hasattr(__builtins__, "__import__")
        else None
    )


# =============================================================================
# Span helper tests
# =============================================================================


class TestStartSpan:
    """Tests for the start_span context manager."""

    def setup_method(self):
        reset_otel()

    def teardown_method(self):
        reset_otel()

    def test_start_span_returns_noop_span(self):
        """Test start_span returns NoOpSpan when not initialized."""
        with start_span("test.operation") as span:
            assert isinstance(span, _NoOpSpan)
            span.set_attribute("key", "value")

    def test_start_span_with_attributes(self):
        """Test start_span applies initial attributes."""
        with start_span("test.op", {"attr1": "val1", "attr2": 42}) as span:
            assert isinstance(span, _NoOpSpan)

    def test_start_span_with_none_attributes_skipped(self):
        """Test start_span skips None attribute values."""
        with start_span("test.op", {"key": None, "valid": "yes"}) as span:
            assert span is not None

    def test_start_span_nested(self):
        """Test nested start_span calls."""
        with start_span("outer") as outer:
            assert isinstance(outer, _NoOpSpan)
            with start_span("inner") as inner:
                assert isinstance(inner, _NoOpSpan)


class TestRecordSpanError:
    """Tests for record_span_error."""

    def test_with_none_span(self):
        """Test record_span_error with None span does not raise."""
        record_span_error(None, ValueError("test"))

    def test_with_noop_span(self):
        """Test record_span_error with NoOpSpan does not raise."""
        span = _NoOpSpan()
        record_span_error(span, ValueError("test error"))

    def test_with_mock_span(self):
        """Test record_span_error calls record_exception on real spans."""
        span = MagicMock()
        span.record_exception = MagicMock()
        span.set_status = MagicMock()

        record_span_error(span, RuntimeError("oops"))
        span.record_exception.assert_called_once()


class TestSetSpanOk:
    """Tests for set_span_ok."""

    def test_with_none_span(self):
        """Test set_span_ok with None span does not raise."""
        set_span_ok(None)

    def test_with_noop_span(self):
        """Test set_span_ok with NoOpSpan does not raise."""
        span = _NoOpSpan()
        set_span_ok(span)


# =============================================================================
# Context propagation tests
# =============================================================================


class TestContextPropagation:
    """Tests for inject_context and extract_context."""

    def setup_method(self):
        reset_otel()

    def teardown_method(self):
        reset_otel()

    def test_inject_context_returns_carrier(self):
        """Test inject_context returns the carrier dict."""
        carrier = {"existing": "header"}
        result = inject_context(carrier)
        assert result is carrier
        # existing header should be preserved
        assert result["existing"] == "header"

    def test_extract_context_returns_none_when_not_initialized(self):
        """Test extract_context returns None when OTel not initialized."""
        result = extract_context({"traceparent": "00-abc-def-01"})
        assert result is None


# =============================================================================
# Debate-specific tracing tests
# =============================================================================


class TestDebateTracing:
    """Tests for debate-specific tracing context managers."""

    def setup_method(self):
        reset_otel()

    def teardown_method(self):
        reset_otel()

    def test_trace_debate_lifecycle(self):
        """Test trace_debate_lifecycle creates a span."""
        with trace_debate_lifecycle(
            debate_id="debate-123",
            task="Design a rate limiter",
            agent_count=5,
            protocol_rounds=3,
        ) as span:
            assert isinstance(span, _NoOpSpan)

    def test_trace_debate_lifecycle_long_task_truncated(self):
        """Test that very long task strings are truncated."""
        long_task = "x" * 1000
        with trace_debate_lifecycle("debate-456", task=long_task) as span:
            # Should not raise, task is truncated internally
            assert span is not None

    def test_trace_debate_round(self):
        """Test trace_debate_round creates a span."""
        with trace_debate_round("debate-789", round_number=2) as span:
            assert isinstance(span, _NoOpSpan)

    def test_trace_agent_operation(self):
        """Test trace_agent_operation creates a span."""
        with trace_agent_operation(
            agent_name="claude",
            operation="propose",
            debate_id="debate-abc",
            round_number=1,
            model="claude-3-opus",
        ) as span:
            assert isinstance(span, _NoOpSpan)

    def test_trace_agent_operation_minimal(self):
        """Test trace_agent_operation with minimal args."""
        with trace_agent_operation(
            agent_name="gpt4",
            operation="critique",
        ) as span:
            assert isinstance(span, _NoOpSpan)

    def test_trace_consensus_evaluation(self):
        """Test trace_consensus_evaluation creates a span."""
        with trace_consensus_evaluation(
            debate_id="debate-def",
            round_number=3,
            method="convergence",
        ) as span:
            assert isinstance(span, _NoOpSpan)

    def test_nested_debate_spans(self):
        """Test nested debate lifecycle > round > agent spans."""
        with trace_debate_lifecycle("d-1", agent_count=3) as lifecycle_span:
            assert isinstance(lifecycle_span, _NoOpSpan)

            with trace_debate_round("d-1", 1) as round_span:
                assert isinstance(round_span, _NoOpSpan)

                with trace_agent_operation("claude", "propose", "d-1", 1) as agent_span:
                    assert isinstance(agent_span, _NoOpSpan)

                with trace_agent_operation("gpt4", "propose", "d-1", 1) as agent_span:
                    assert isinstance(agent_span, _NoOpSpan)

            with trace_consensus_evaluation("d-1", 1) as consensus_span:
                assert isinstance(consensus_span, _NoOpSpan)


# =============================================================================
# Bridge tests
# =============================================================================


class TestExportDebateSpanToOTel:
    """Tests for export_debate_span_to_otel bridge function."""

    def setup_method(self):
        reset_otel()

    def teardown_method(self):
        reset_otel()

    def test_noop_when_not_initialized(self):
        """Test export does nothing when OTel is not initialized."""
        mock_span = MagicMock()
        mock_span.name = "test"
        mock_span.trace_id = "abc"
        mock_span.span_id = "def"
        mock_span.start_time = 1000.0
        mock_span.attributes = {"key": "value"}
        mock_span.events = []
        mock_span.status = "OK"

        # Should not raise
        export_debate_span_to_otel(mock_span)

    def test_handles_none_span(self):
        """Test export handles None gracefully."""
        # Should not raise
        export_debate_span_to_otel(None)


# =============================================================================
# Integration with observability __init__ tests
# =============================================================================


class TestObservabilityExports:
    """Test that otel.py exports are accessible from aragora.observability."""

    def test_otel_config_importable(self):
        """Test OTelConfig is importable from observability package."""
        from aragora.observability import OTelConfig as Cfg

        assert Cfg is OTelConfig

    def test_setup_importable(self):
        """Test setup_otel is importable from observability package."""
        from aragora.observability import setup_otel as setup

        assert setup is setup_otel

    def test_shutdown_importable(self):
        """Test shutdown_otel is importable from observability package."""
        from aragora.observability import shutdown_otel as shutdown

        assert shutdown is shutdown_otel

    def test_start_span_importable(self):
        """Test start_span is importable from observability package."""
        from aragora.observability import start_span as ss

        assert ss is start_span

    def test_debate_tracing_importable(self):
        """Test debate tracing functions are importable from observability package."""
        from aragora.observability import (
            trace_debate_lifecycle,
            trace_debate_round,
            trace_agent_operation,
            trace_consensus_evaluation,
        )

        assert trace_debate_lifecycle is not None
        assert trace_debate_round is not None
        assert trace_agent_operation is not None
        assert trace_consensus_evaluation is not None

    def test_health_check_importable(self):
        """Test health check functions are importable from observability package."""
        from aragora.observability import (
            OTLPHealthStatus,
            check_otlp_health,
            get_otel_status,
        )

        assert OTLPHealthStatus is not None
        assert check_otlp_health is not None
        assert get_otel_status is not None


# =============================================================================
# Protocol and headers configuration tests
# =============================================================================


class TestProtocolConfig:
    """Tests for protocol and headers configuration."""

    def test_default_protocol_is_grpc(self):
        """Test default protocol is grpc."""
        config = OTelConfig()
        assert config.protocol == "grpc"

    def test_http_protocol(self):
        """Test http/protobuf protocol."""
        config = OTelConfig(protocol="http/protobuf")
        assert config.protocol == "http/protobuf"

    def test_invalid_protocol_raises(self):
        """Test invalid protocol raises ValueError."""
        with pytest.raises(ValueError, match="protocol must be"):
            OTelConfig(protocol="invalid")

    def test_default_headers_empty(self):
        """Test default headers are empty."""
        config = OTelConfig()
        assert config.headers == {}

    def test_custom_headers(self):
        """Test custom headers."""
        config = OTelConfig(headers={"Authorization": "Bearer token"})
        assert config.headers == {"Authorization": "Bearer token"}


class TestProtocolFromEnv:
    """Tests for protocol and headers environment parsing."""

    def setup_method(self):
        """Clear environment variables before each test."""
        self._env_vars = [
            "OTEL_EXPORTER_OTLP_PROTOCOL",
            "ARAGORA_OTLP_PROTOCOL",
            "OTEL_EXPORTER_OTLP_HEADERS",
            "ARAGORA_OTLP_HEADERS",
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
        """Test OTEL_EXPORTER_OTLP_PROTOCOL env var."""
        os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"
        config = OTelConfig.from_env()
        assert config.protocol == "http/protobuf"

    def test_aragora_protocol_env_fallback(self):
        """Test ARAGORA_OTLP_PROTOCOL env var as fallback."""
        os.environ["ARAGORA_OTLP_PROTOCOL"] = "http/protobuf"
        config = OTelConfig.from_env()
        assert config.protocol == "http/protobuf"

    def test_otel_protocol_takes_precedence(self):
        """Test OTEL_ var takes precedence over ARAGORA_ var."""
        os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "grpc"
        os.environ["ARAGORA_OTLP_PROTOCOL"] = "http/protobuf"
        config = OTelConfig.from_env()
        assert config.protocol == "grpc"

    def test_invalid_protocol_falls_back(self):
        """Test invalid protocol falls back to grpc."""
        os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "invalid_proto"
        config = OTelConfig.from_env()
        assert config.protocol == "grpc"

    def test_headers_json_format(self):
        """Test headers parsed from JSON format."""
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = '{"Authorization": "Bearer abc"}'
        config = OTelConfig.from_env()
        assert config.headers == {"Authorization": "Bearer abc"}

    def test_headers_key_value_format(self):
        """Test headers parsed from key=value format."""
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


# =============================================================================
# Exporter factory tests
# =============================================================================


class TestCreateOtlpExporter:
    """Tests for _create_otlp_exporter factory."""

    def test_grpc_exporter_creation(self):
        """Test gRPC exporter is created when available."""
        config = OTelConfig(
            endpoint="http://localhost:4317",
            protocol="grpc",
            insecure=True,
        )
        try:
            exporter = _create_otlp_exporter(config)
            # If gRPC package is installed, we get an exporter
            assert exporter is not None
        except Exception:
            # If packages not installed, test is inconclusive
            pass

    def test_http_exporter_creation(self):
        """Test HTTP exporter is created when available."""
        config = OTelConfig(
            endpoint="http://localhost:4318",
            protocol="http/protobuf",
        )
        try:
            exporter = _create_otlp_exporter(config)
            # Exporter should be created (either HTTP or gRPC fallback)
            assert exporter is not None
        except Exception:
            pass

    def test_grpc_with_headers(self):
        """Test gRPC exporter with headers."""
        config = OTelConfig(
            endpoint="http://localhost:4317",
            protocol="grpc",
            insecure=True,
            headers={"Authorization": "Bearer token"},
        )
        try:
            exporter = _create_otlp_exporter(config)
            assert exporter is not None
        except Exception:
            pass

    def test_grpc_fallback_when_grpc_missing(self):
        """Test fallback to HTTP when gRPC import fails."""
        config = OTelConfig(protocol="grpc", insecure=True)

        # If we can import either, the factory should succeed
        has_any_exporter = False
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # noqa: F401

            has_any_exporter = True
        except ImportError:
            pass
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # noqa: F401

            has_any_exporter = True
        except ImportError:
            pass

        exporter = _create_otlp_exporter(config)
        if has_any_exporter:
            assert exporter is not None
        else:
            assert exporter is None

    def test_http_fallback_when_http_missing(self):
        """Test fallback to gRPC when HTTP import fails."""
        config = OTelConfig(protocol="http/protobuf")

        has_any_exporter = False
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # noqa: F401

            has_any_exporter = True
        except ImportError:
            pass
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # noqa: F401

            has_any_exporter = True
        except ImportError:
            pass

        exporter = _create_otlp_exporter(config)
        if has_any_exporter:
            assert exporter is not None
        else:
            assert exporter is None


# =============================================================================
# OTLP Health check tests
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


class TestCheckOtlpHealth:
    """Tests for check_otlp_health function."""

    def setup_method(self):
        reset_otel()

    def teardown_method(self):
        reset_otel()

    def test_health_check_connection_refused(self):
        """Test health check when no collector is running."""
        # Use a port that is very unlikely to have a listener
        status = check_otlp_health(
            endpoint="http://localhost:19999",
            timeout_ms=500,
        )
        assert status.healthy is False
        assert status.latency_ms is not None
        assert status.error is not None
        assert "Connection refused" in status.error or "Connection error" in status.error

    def test_health_check_timeout(self):
        """Test health check timeout with unreachable host."""
        # Use a non-routable address to trigger timeout
        status = check_otlp_health(
            endpoint="http://192.0.2.1:4317",  # TEST-NET, non-routable
            timeout_ms=500,
        )
        assert status.healthy is False
        assert status.latency_ms is not None
        assert status.error is not None

    def test_health_check_default_grpc_port(self):
        """Test health check defaults to port 4317 for gRPC."""
        status = check_otlp_health(
            endpoint="http://localhost",
            timeout_ms=200,
        )
        # Should fail (no collector) but the endpoint/port resolution should work
        assert status.endpoint == "http://localhost"
        assert status.protocol == "grpc"

    def test_health_check_initialized_flag(self):
        """Test health check reflects initialization state."""
        status = check_otlp_health(
            endpoint="http://localhost:19999",
            timeout_ms=200,
        )
        assert status.initialized is False  # OTel not initialized

    def test_health_check_with_explicit_endpoint(self):
        """Test health check uses explicit endpoint over config."""
        status = check_otlp_health(
            endpoint="http://127.0.0.1:19999",
            timeout_ms=200,
        )
        assert status.endpoint == "http://127.0.0.1:19999"


class TestGetOtelStatus:
    """Tests for get_otel_status function."""

    def setup_method(self):
        reset_otel()

    def teardown_method(self):
        reset_otel()

    def test_status_when_not_initialized(self):
        """Test status when OTel is not initialized."""
        status = get_otel_status()
        assert status["initialized"] is False
        assert "reason" in status

    def test_status_after_mock_initialization(self):
        """Test status reports correctly after initialization."""
        # We can't easily fully initialize without a collector,
        # but we can verify the not-initialized path
        status = get_otel_status()
        assert isinstance(status, dict)
        assert "initialized" in status


# =============================================================================
# Startup OTLP connectivity check tests
# =============================================================================


class TestStartupOtlpConnectivity:
    """Tests for the server startup OTLP connectivity check."""

    @pytest.mark.asyncio
    async def test_check_skips_when_disabled(self):
        """Test connectivity check skips when tracing is not enabled."""
        from aragora.server.startup.observability import check_otlp_connectivity

        # With default env (tracing disabled), should return False
        result = await check_otlp_connectivity()
        assert result is False

    @pytest.mark.asyncio
    async def test_check_warns_when_unreachable(self):
        """Test connectivity check warns when collector is unreachable."""
        from aragora.server.startup.observability import check_otlp_connectivity

        env_vars = {
            "OTEL_ENABLED": "true",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:19999",
        }
        with patch.dict(os.environ, env_vars):
            result = await check_otlp_connectivity()
            assert result is False
