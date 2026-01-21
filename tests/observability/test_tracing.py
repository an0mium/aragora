"""
Tests for OpenTelemetry distributed tracing module.
"""

import pytest


class TestNoOpTracer:
    """Tests for no-op tracer behavior when OpenTelemetry is not installed or disabled."""

    def test_noop_tracer_returns_noop_span(self):
        """Test that no-op tracer returns no-op spans."""
        from aragora.observability.tracing import _NoOpTracer

        tracer = _NoOpTracer()
        span = tracer.start_as_current_span("test")

        assert span is not None
        # Should not raise
        span.set_attribute("key", "value")
        span.set_attributes({"a": 1, "b": 2})
        span.add_event("event", {"attr": "val"})
        span.record_exception(ValueError("test"))
        span.set_status(None)
        span.end()

    def test_noop_span_context_manager(self):
        """Test that no-op span works as context manager."""
        from aragora.observability.tracing import _NoOpTracer

        tracer = _NoOpTracer()
        with tracer.start_as_current_span("test") as span:
            span.set_attribute("key", "value")
            # Should not raise


class TestTracingInitialization:
    """Tests for tracer initialization."""

    def test_get_tracer_returns_tracer(self):
        """Test that get_tracer returns a tracer (may be no-op)."""
        from aragora.observability.tracing import get_tracer

        tracer = get_tracer()
        assert tracer is not None
        # Should have start_as_current_span method
        assert hasattr(tracer, "start_as_current_span")

    def test_create_span_context_manager(self):
        """Test create_span context manager."""
        from aragora.observability.tracing import create_span

        with create_span("test.span", {"attr": "value"}) as span:
            assert span is not None


class TestHandlerTracing:
    """Tests for handler tracing decorators."""

    def test_trace_handler_decorator(self):
        """Test trace_handler decorator."""
        from aragora.observability.tracing import trace_handler

        @trace_handler("test.handler")
        def handle_request(self, handler):
            return {"status": "ok"}

        class MockSelf:
            pass

        class MockHandler:
            path = "/test"
            command = "GET"
            client_address = ("127.0.0.1", 8080)

        result = handle_request(MockSelf(), MockHandler())
        assert result == {"status": "ok"}

    def test_trace_handler_with_exception(self):
        """Test trace_handler records exceptions."""
        from aragora.observability.tracing import trace_handler

        @trace_handler("test.failing_handler")
        def handle_failing(self, handler):
            raise ValueError("Test error")

        class MockSelf:
            pass

        class MockHandler:
            path = "/test"
            command = "POST"
            client_address = ("127.0.0.1", 8080)

        with pytest.raises(ValueError, match="Test error"):
            handle_failing(MockSelf(), MockHandler())


class TestWebhookTracing:
    """Tests for webhook-specific tracing."""

    def test_trace_webhook_delivery(self):
        """Test trace_webhook_delivery context manager."""
        from aragora.observability.tracing import trace_webhook_delivery

        with trace_webhook_delivery(
            event_type="slo_violation",
            webhook_id="webhook-123",
            webhook_url="https://example.com/webhook",
            correlation_id="corr-456",
        ) as span:
            assert span is not None
            span.set_attribute("webhook.success", True)
            span.set_attribute("webhook.status_code", 200)

    def test_trace_webhook_delivery_without_correlation_id(self):
        """Test trace_webhook_delivery without correlation ID."""
        from aragora.observability.tracing import trace_webhook_delivery

        with trace_webhook_delivery(
            event_type="debate_end",
            webhook_id="webhook-789",
            webhook_url="https://example.com/hook",
        ) as span:
            assert span is not None

    def test_trace_webhook_batch(self):
        """Test trace_webhook_batch context manager."""
        from aragora.observability.tracing import trace_webhook_batch

        with trace_webhook_batch(
            event_type="slo_violation",
            batch_size=10,
            correlation_id="batch-123",
        ) as span:
            assert span is not None
            span.set_attribute("webhook.batch_success", True)

    def test_trace_webhook_batch_without_correlation_id(self):
        """Test trace_webhook_batch without correlation ID."""
        from aragora.observability.tracing import trace_webhook_batch

        with trace_webhook_batch(
            event_type="debate_update",
            batch_size=5,
        ) as span:
            assert span is not None

    def test_redact_url(self):
        """Test URL redaction for tracing."""
        from aragora.observability.tracing import _redact_url

        # Query params should be removed
        url = "https://example.com/webhook?secret=abc123&token=xyz"
        redacted = _redact_url(url)
        assert "secret" not in redacted
        assert "token" not in redacted
        assert "example.com/webhook" in redacted

        # Fragment should be removed
        url = "https://example.com/hook#fragment"
        redacted = _redact_url(url)
        assert "fragment" not in redacted

        # Basic URL should be preserved
        url = "https://example.com/webhook/endpoint"
        redacted = _redact_url(url)
        assert redacted == "https://example.com/webhook/endpoint"


class TestDebateTracing:
    """Tests for debate-specific tracing."""

    def test_trace_debate_phase(self):
        """Test trace_debate_phase context manager."""
        from aragora.observability.tracing import trace_debate_phase

        with trace_debate_phase("propose", "debate-123", round_num=1) as span:
            assert span is not None

    def test_trace_consensus_check(self):
        """Test trace_consensus_check context manager."""
        from aragora.observability.tracing import trace_consensus_check

        with trace_consensus_check("debate-456", round_num=2) as span:
            assert span is not None


class TestDecisionTracing:
    """Tests for decision routing tracing."""

    def test_trace_decision_routing(self):
        """Test trace_decision_routing context manager."""
        from aragora.observability.tracing import trace_decision_routing

        with trace_decision_routing(
            request_id="req-123",
            decision_type="debate",
            source="http_api",
            priority="high",
        ) as span:
            assert span is not None

    def test_trace_decision_engine(self):
        """Test trace_decision_engine context manager."""
        from aragora.observability.tracing import trace_decision_engine

        with trace_decision_engine("debate", "req-456") as span:
            assert span is not None

    def test_trace_response_delivery(self):
        """Test trace_response_delivery context manager."""
        from aragora.observability.tracing import trace_response_delivery

        with trace_response_delivery(
            platform="slack",
            channel_id="C123456",
            voice_enabled=False,
        ) as span:
            assert span is not None


class TestSpanAttributes:
    """Tests for span attribute helpers."""

    def test_add_span_attributes(self):
        """Test add_span_attributes helper."""
        from aragora.observability.tracing import add_span_attributes, _NoOpSpan

        span = _NoOpSpan()
        # Should not raise
        add_span_attributes(span, {"key": "value", "count": 42, "none_val": None})

    def test_add_span_attributes_with_none_span(self):
        """Test add_span_attributes with None span."""
        from aragora.observability.tracing import add_span_attributes

        # Should not raise
        add_span_attributes(None, {"key": "value"})

    def test_record_exception(self):
        """Test record_exception helper."""
        from aragora.observability.tracing import record_exception, _NoOpSpan

        span = _NoOpSpan()
        # Should not raise
        record_exception(span, ValueError("test"))

    def test_record_exception_with_none_span(self):
        """Test record_exception with None span."""
        from aragora.observability.tracing import record_exception

        # Should not raise
        record_exception(None, ValueError("test"))
