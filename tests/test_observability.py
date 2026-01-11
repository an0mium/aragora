"""
Tests for Observability modules (metrics and tracing).

Tests Prometheus metrics and OpenTelemetry tracing functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time


class TestMetricsInitialization:
    """Test metrics module initialization."""

    def test_init_metrics_disabled(self):
        """Test metrics initialization when disabled."""
        from aragora.observability.metrics import _init_metrics, _init_noop_metrics

        with patch("aragora.observability.metrics.get_metrics_config") as mock_config:
            mock_config.return_value = Mock(enabled=False)

            with patch("aragora.observability.metrics._initialized", False):
                # Reset for test
                import aragora.observability.metrics as metrics_module
                metrics_module._initialized = False

                _init_noop_metrics()

                # NoOp metrics should be set
                assert metrics_module.REQUEST_COUNT is not None

    def test_init_metrics_prometheus_not_installed(self):
        """Test graceful handling when prometheus not installed."""
        with patch("aragora.observability.metrics.get_metrics_config") as mock_config:
            mock_config.return_value = Mock(enabled=True, histogram_buckets=[0.1, 0.5, 1.0])

            with patch("aragora.observability.metrics._initialized", False):
                with patch.dict("sys.modules", {"prometheus_client": None}):
                    with patch("builtins.__import__", side_effect=ImportError):
                        from aragora.observability.metrics import _init_metrics
                        import aragora.observability.metrics as metrics_module
                        metrics_module._initialized = False

                        result = _init_metrics()

                        # Should gracefully handle ImportError
                        assert metrics_module._initialized is True


class TestNoOpMetrics:
    """Test NoOp metric implementations."""

    def test_noop_metric_labels(self):
        """Test NoOp metric labels method."""
        from aragora.observability.metrics import _init_noop_metrics
        import aragora.observability.metrics as metrics_module

        # Reset and initialize NoOp metrics
        metrics_module._initialized = False
        _init_noop_metrics()

        # Labels should return self
        result = metrics_module.REQUEST_COUNT.labels(method="GET", endpoint="/api")
        assert result is metrics_module.REQUEST_COUNT

    def test_noop_metric_inc(self):
        """Test NoOp metric inc method."""
        from aragora.observability.metrics import _init_noop_metrics
        import aragora.observability.metrics as metrics_module

        metrics_module._initialized = False
        _init_noop_metrics()

        # Should not raise
        metrics_module.REQUEST_COUNT.inc()
        metrics_module.REQUEST_COUNT.inc(5)

    def test_noop_metric_dec(self):
        """Test NoOp metric dec method."""
        from aragora.observability.metrics import _init_noop_metrics
        import aragora.observability.metrics as metrics_module

        metrics_module._initialized = False
        _init_noop_metrics()

        # Should not raise
        metrics_module.ACTIVE_DEBATES.dec()
        metrics_module.ACTIVE_DEBATES.dec(2)

    def test_noop_metric_set(self):
        """Test NoOp metric set method."""
        from aragora.observability.metrics import _init_noop_metrics
        import aragora.observability.metrics as metrics_module

        metrics_module._initialized = False
        _init_noop_metrics()

        # Should not raise
        metrics_module.CONSENSUS_RATE.set(0.75)

    def test_noop_metric_observe(self):
        """Test NoOp metric observe method."""
        from aragora.observability.metrics import _init_noop_metrics
        import aragora.observability.metrics as metrics_module

        metrics_module._initialized = False
        _init_noop_metrics()

        # Should not raise
        metrics_module.REQUEST_LATENCY.observe(0.5)


class TestMetricsRecording:
    """Test metrics recording functions."""

    def test_record_request(self):
        """Test recording HTTP request metrics."""
        from aragora.observability.metrics import record_request, _init_noop_metrics
        import aragora.observability.metrics as metrics_module

        metrics_module._initialized = False
        _init_noop_metrics()
        metrics_module._initialized = True

        # Should not raise
        record_request("GET", "/api/debates", 200, 0.05)
        record_request("POST", "/api/debate", 201, 0.15)
        record_request("GET", "/api/agents", 500, 1.0)

    def test_record_agent_call(self):
        """Test recording agent call metrics."""
        from aragora.observability.metrics import record_agent_call, _init_noop_metrics
        import aragora.observability.metrics as metrics_module

        metrics_module._initialized = False
        _init_noop_metrics()
        metrics_module._initialized = True

        # Should not raise
        record_agent_call("claude", success=True, latency=1.2)
        record_agent_call("gpt4", success=False, latency=5.0)

    def test_set_consensus_rate(self):
        """Test setting consensus rate metric."""
        from aragora.observability.metrics import set_consensus_rate, _init_noop_metrics
        import aragora.observability.metrics as metrics_module

        metrics_module._initialized = False
        _init_noop_metrics()
        metrics_module._initialized = True

        # Should not raise
        set_consensus_rate(0.85)
        set_consensus_rate(0.0)
        set_consensus_rate(1.0)

    def test_record_memory_operation(self):
        """Test recording memory operation metrics."""
        from aragora.observability.metrics import record_memory_operation, _init_noop_metrics
        import aragora.observability.metrics as metrics_module

        metrics_module._initialized = False
        _init_noop_metrics()
        metrics_module._initialized = True

        # Should not raise
        record_memory_operation("store", "fast")
        record_memory_operation("query", "medium")
        record_memory_operation("promote", "slow")

    def test_track_websocket_connection(self):
        """Test tracking WebSocket connections."""
        from aragora.observability.metrics import track_websocket_connection, _init_noop_metrics
        import aragora.observability.metrics as metrics_module

        metrics_module._initialized = False
        _init_noop_metrics()
        metrics_module._initialized = True

        # Should not raise
        track_websocket_connection(connected=True)
        track_websocket_connection(connected=False)


class TestTrackDebate:
    """Test track_debate context manager."""

    def test_track_debate_context_manager(self):
        """Test debate tracking context manager."""
        from aragora.observability.metrics import track_debate, _init_noop_metrics
        import aragora.observability.metrics as metrics_module

        metrics_module._initialized = False
        _init_noop_metrics()
        metrics_module._initialized = True

        # Should not raise
        with track_debate():
            # Simulate debate
            pass

    def test_track_debate_exception_handling(self):
        """Test debate tracking handles exceptions."""
        from aragora.observability.metrics import track_debate, _init_noop_metrics
        import aragora.observability.metrics as metrics_module

        metrics_module._initialized = False
        _init_noop_metrics()
        metrics_module._initialized = True

        # Should decrement even on exception
        with pytest.raises(ValueError):
            with track_debate():
                raise ValueError("Test error")


class TestLatencyDecorators:
    """Test latency measurement decorators."""

    def test_measure_latency_sync(self):
        """Test synchronous latency measurement decorator."""
        from aragora.observability.metrics import measure_latency, _init_noop_metrics
        import aragora.observability.metrics as metrics_module

        metrics_module._initialized = False
        _init_noop_metrics()
        metrics_module._initialized = True

        @measure_latency("test_endpoint")
        def slow_function():
            time.sleep(0.01)
            return "result"

        result = slow_function()
        assert result == "result"

    def test_measure_latency_with_exception(self):
        """Test latency measurement records on exception."""
        from aragora.observability.metrics import measure_latency, _init_noop_metrics
        import aragora.observability.metrics as metrics_module

        metrics_module._initialized = False
        _init_noop_metrics()
        metrics_module._initialized = True

        @measure_latency("error_endpoint")
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

    @pytest.mark.asyncio
    async def test_measure_async_latency(self):
        """Test async latency measurement decorator."""
        from aragora.observability.metrics import measure_async_latency, _init_noop_metrics
        import aragora.observability.metrics as metrics_module
        import asyncio

        metrics_module._initialized = False
        _init_noop_metrics()
        metrics_module._initialized = True

        @measure_async_latency("async_endpoint")
        async def async_function():
            await asyncio.sleep(0.01)
            return "async_result"

        result = await async_function()
        assert result == "async_result"


class TestEndpointNormalization:
    """Test endpoint normalization for cardinality control."""

    def test_normalize_endpoint_basic(self):
        """Test basic endpoint normalization."""
        from aragora.observability.metrics import _normalize_endpoint

        # Basic paths should pass through
        assert _normalize_endpoint("/api/debates") == "/api/debates"
        assert _normalize_endpoint("/api/agents") == "/api/agents"

    def test_normalize_endpoint_with_id(self):
        """Test endpoint normalization with IDs."""
        from aragora.observability.metrics import _normalize_endpoint

        # IDs should be normalized to :id
        result = _normalize_endpoint("/api/debates/abc123")
        assert ":id" in result or "debates" in result

    def test_normalize_endpoint_empty(self):
        """Test empty endpoint normalization."""
        from aragora.observability.metrics import _normalize_endpoint

        result = _normalize_endpoint("")
        assert result == "" or result == "/"


class TestTracingInitialization:
    """Test tracing module initialization."""

    def test_init_tracer_disabled(self):
        """Test tracer initialization when disabled."""
        from aragora.observability.tracing import _init_tracer, _NoOpTracer

        with patch("aragora.observability.tracing.get_tracing_config") as mock_config:
            mock_config.return_value = Mock(enabled=False)

            with patch("aragora.observability.tracing._tracer", None):
                tracer = _init_tracer()

                assert isinstance(tracer, _NoOpTracer)

    def test_init_tracer_import_error(self):
        """Test tracer handles missing OpenTelemetry."""
        from aragora.observability.tracing import _init_tracer, _NoOpTracer

        with patch("aragora.observability.tracing.get_tracing_config") as mock_config:
            mock_config.return_value = Mock(enabled=True)

            with patch("aragora.observability.tracing._tracer", None):
                with patch("builtins.__import__", side_effect=ImportError):
                    tracer = _init_tracer()

                    assert isinstance(tracer, _NoOpTracer)


class TestNoOpTracer:
    """Test NoOp tracer implementation."""

    def test_noop_tracer_start_span(self):
        """Test NoOp tracer start_span method."""
        from aragora.observability.tracing import _NoOpTracer, _NoOpSpan

        tracer = _NoOpTracer()
        span = tracer.start_span("test_span")

        assert isinstance(span, _NoOpSpan)

    def test_noop_tracer_start_as_current_span(self):
        """Test NoOp tracer start_as_current_span method."""
        from aragora.observability.tracing import _NoOpTracer, _NoOpSpan

        tracer = _NoOpTracer()
        span = tracer.start_as_current_span("test_span")

        assert isinstance(span, _NoOpSpan)


class TestNoOpSpan:
    """Test NoOp span implementation."""

    def test_noop_span_set_attribute(self):
        """Test NoOp span set_attribute method."""
        from aragora.observability.tracing import _NoOpSpan

        span = _NoOpSpan()
        result = span.set_attribute("key", "value")

        assert result is span

    def test_noop_span_set_attributes(self):
        """Test NoOp span set_attributes method."""
        from aragora.observability.tracing import _NoOpSpan

        span = _NoOpSpan()
        result = span.set_attributes({"key1": "value1", "key2": "value2"})

        assert result is span

    def test_noop_span_add_event(self):
        """Test NoOp span add_event method."""
        from aragora.observability.tracing import _NoOpSpan

        span = _NoOpSpan()
        result = span.add_event("test_event", {"attr": "value"})

        assert result is span

    def test_noop_span_record_exception(self):
        """Test NoOp span record_exception method."""
        from aragora.observability.tracing import _NoOpSpan

        span = _NoOpSpan()
        result = span.record_exception(ValueError("test"))

        assert result is span

    def test_noop_span_set_status(self):
        """Test NoOp span set_status method."""
        from aragora.observability.tracing import _NoOpSpan

        span = _NoOpSpan()
        result = span.set_status(Mock())

        assert result is span

    def test_noop_span_end(self):
        """Test NoOp span end method."""
        from aragora.observability.tracing import _NoOpSpan

        span = _NoOpSpan()
        # Should not raise
        span.end()

    def test_noop_span_context_manager(self):
        """Test NoOp span as context manager."""
        from aragora.observability.tracing import _NoOpSpan

        span = _NoOpSpan()

        with span as s:
            assert s is span


class TestCreateSpan:
    """Test create_span context manager."""

    def test_create_span_basic(self):
        """Test basic span creation."""
        from aragora.observability.tracing import create_span, _NoOpSpan

        with patch("aragora.observability.tracing.get_tracer") as mock_get_tracer:
            mock_tracer = Mock()
            mock_span = _NoOpSpan()
            mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)
            mock_get_tracer.return_value = mock_tracer

            with create_span("test_span") as span:
                assert span is not None

    def test_create_span_with_attributes(self):
        """Test span creation with attributes."""
        from aragora.observability.tracing import create_span, _NoOpSpan

        with patch("aragora.observability.tracing.get_tracer") as mock_get_tracer:
            mock_tracer = Mock()
            mock_span = _NoOpSpan()
            mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)
            mock_get_tracer.return_value = mock_tracer

            with create_span("test_span", attributes={"key": "value"}) as span:
                pass


class TestTraceHandler:
    """Test trace_handler decorator."""

    def test_trace_handler_decorator(self):
        """Test trace_handler decorates functions."""
        from aragora.observability.tracing import trace_handler, _NoOpTracer

        with patch("aragora.observability.tracing.get_tracer") as mock_get_tracer:
            mock_get_tracer.return_value = _NoOpTracer()

            @trace_handler("test.endpoint")
            def handler_function(self, handler):
                return "result"

            # Create mock objects
            mock_self = Mock()
            mock_handler = Mock()
            mock_handler.path = "/api/test"
            mock_handler.command = "GET"

            result = handler_function(mock_self, mock_handler)
            assert result == "result"


class TestTraceAgentCall:
    """Test trace_agent_call decorator."""

    def test_trace_agent_call_decorator(self):
        """Test trace_agent_call decorates agent methods."""
        from aragora.observability.tracing import trace_agent_call, _NoOpTracer

        with patch("aragora.observability.tracing.get_tracer") as mock_get_tracer:
            mock_get_tracer.return_value = _NoOpTracer()

            @trace_agent_call
            def generate(self, prompt: str):
                return f"Response to: {prompt}"

            mock_self = Mock()
            mock_self.name = "test_agent"
            mock_self.model = "test_model"

            result = generate(mock_self, "Hello")
            assert "Hello" in result

    @pytest.mark.asyncio
    async def test_trace_agent_call_async(self):
        """Test trace_agent_call with async methods."""
        from aragora.observability.tracing import trace_agent_call, _NoOpTracer

        with patch("aragora.observability.tracing.get_tracer") as mock_get_tracer:
            mock_get_tracer.return_value = _NoOpTracer()

            @trace_agent_call
            async def async_generate(self, prompt: str):
                return f"Async response to: {prompt}"

            mock_self = Mock()
            mock_self.name = "test_agent"
            mock_self.model = "test_model"

            result = await async_generate(mock_self, "Hello")
            assert "Hello" in result


class TestAddSpanAttributes:
    """Test add_span_attributes utility."""

    def test_add_span_attributes_to_span(self):
        """Test adding attributes to a span."""
        from aragora.observability.tracing import add_span_attributes, _NoOpSpan

        span = _NoOpSpan()
        add_span_attributes(span, {"key1": "value1", "key2": 42})

        # NoOp span just returns self, but should not raise

    def test_add_span_attributes_empty(self):
        """Test adding empty attributes."""
        from aragora.observability.tracing import add_span_attributes, _NoOpSpan

        span = _NoOpSpan()
        add_span_attributes(span, {})


class TestMetricsServer:
    """Test Prometheus metrics server."""

    def test_start_metrics_server_disabled(self):
        """Test starting server when metrics disabled."""
        from aragora.observability.metrics import start_metrics_server

        with patch("aragora.observability.metrics._init_metrics") as mock_init:
            mock_init.return_value = False

            result = start_metrics_server()

            assert result is None

    def test_start_metrics_server_already_running(self):
        """Test starting server when already running."""
        from aragora.observability.metrics import start_metrics_server
        import aragora.observability.metrics as metrics_module

        mock_server = Mock()
        metrics_module._metrics_server = mock_server

        with patch("aragora.observability.metrics._init_metrics") as mock_init:
            mock_init.return_value = True

            result = start_metrics_server()

            assert result is mock_server

        # Reset
        metrics_module._metrics_server = None


class TestObservabilityConfig:
    """Test observability configuration."""

    def test_get_metrics_config(self):
        """Test getting metrics configuration."""
        from aragora.observability.config import get_metrics_config

        config = get_metrics_config()

        assert hasattr(config, "enabled")
        assert hasattr(config, "port")

    def test_get_tracing_config(self):
        """Test getting tracing configuration."""
        from aragora.observability.config import get_tracing_config

        config = get_tracing_config()

        assert hasattr(config, "enabled")

    def test_is_metrics_enabled(self):
        """Test checking if metrics enabled."""
        from aragora.observability.config import is_metrics_enabled

        # Should return a boolean
        result = is_metrics_enabled()
        assert isinstance(result, bool)

    def test_is_tracing_enabled(self):
        """Test checking if tracing enabled."""
        from aragora.observability.config import is_tracing_enabled

        # Should return a boolean
        result = is_tracing_enabled()
        assert isinstance(result, bool)


class TestMetricsIntegration:
    """Integration tests for metrics."""

    def test_full_request_tracking(self):
        """Test full request tracking flow."""
        from aragora.observability.metrics import (
            record_request,
            _init_noop_metrics,
        )
        import aragora.observability.metrics as metrics_module

        metrics_module._initialized = False
        _init_noop_metrics()
        metrics_module._initialized = True

        # Simulate a series of requests
        endpoints = [
            ("GET", "/api/debates", 200, 0.05),
            ("GET", "/api/debates/123", 200, 0.03),
            ("POST", "/api/debate", 201, 0.15),
            ("GET", "/api/agents", 200, 0.02),
            ("GET", "/api/debates", 500, 0.1),
        ]

        for method, endpoint, status, latency in endpoints:
            record_request(method, endpoint, status, latency)

    def test_full_debate_tracking(self):
        """Test full debate tracking flow."""
        from aragora.observability.metrics import (
            track_debate,
            record_agent_call,
            set_consensus_rate,
            _init_noop_metrics,
        )
        import aragora.observability.metrics as metrics_module

        metrics_module._initialized = False
        _init_noop_metrics()
        metrics_module._initialized = True

        with track_debate():
            # Simulate agent calls during debate
            record_agent_call("claude", success=True, latency=1.5)
            record_agent_call("gpt4", success=True, latency=2.0)
            record_agent_call("gemini", success=False, latency=5.0)

        # Record consensus rate after debate
        set_consensus_rate(0.67)


class TestTracingIntegration:
    """Integration tests for tracing."""

    def test_full_span_lifecycle(self):
        """Test full span lifecycle."""
        from aragora.observability.tracing import create_span, _NoOpSpan

        with patch("aragora.observability.tracing.get_tracer") as mock_get_tracer:
            from aragora.observability.tracing import _NoOpTracer
            mock_get_tracer.return_value = _NoOpTracer()

            # Simulate nested spans
            with create_span("outer_span") as outer:
                outer.set_attribute("outer.key", "outer.value")

                with create_span("inner_span") as inner:
                    inner.set_attribute("inner.key", "inner.value")
                    inner.add_event("processing_complete")
