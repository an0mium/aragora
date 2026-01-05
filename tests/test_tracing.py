"""Tests for debate tracing module."""

import pytest
import time
from aragora.debate.tracing import (
    Tracer,
    Span,
    SpanContext,
    SpanRecorder,
    get_tracer,
    set_tracer,
    set_debate_context,
    get_debate_context,
    get_debate_id,
    get_metrics,
    clear_metrics,
    generate_trace_id,
    generate_span_id,
    DebateMetrics,
)


class TestTraceIdGeneration:
    """Test trace and span ID generation."""

    def test_generate_trace_id_format(self):
        """Test that trace ID is 32-char hex."""
        trace_id = generate_trace_id()
        assert len(trace_id) == 32
        assert all(c in "0123456789abcdef" for c in trace_id)

    def test_generate_trace_id_unique(self):
        """Test that trace IDs are unique."""
        ids = [generate_trace_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_generate_span_id_format(self):
        """Test that span ID is 16-char hex."""
        span_id = generate_span_id()
        assert len(span_id) == 16
        assert all(c in "0123456789abcdef" for c in span_id)


class TestSpan:
    """Test Span class."""

    def test_span_creation(self):
        """Test basic span creation."""
        span = Span(
            name="test.span",
            trace_id="abc123",
            span_id="def456",
            parent_span_id=None,
            start_time=time.time(),
        )
        assert span.name == "test.span"
        assert span.trace_id == "abc123"
        assert span.status == "OK"

    def test_span_set_attribute(self):
        """Test setting attributes."""
        span = Span("test", "t1", "s1", None, time.time())
        span.set_attribute("key", "value")
        span.set_attribute("count", 42)
        assert span.attributes["key"] == "value"
        assert span.attributes["count"] == 42

    def test_span_set_attributes(self):
        """Test setting multiple attributes."""
        span = Span("test", "t1", "s1", None, time.time())
        span.set_attributes({"a": 1, "b": 2})
        assert span.attributes["a"] == 1
        assert span.attributes["b"] == 2

    def test_span_add_event(self):
        """Test adding events."""
        span = Span("test", "t1", "s1", None, time.time())
        span.add_event("checkpoint", {"progress": 50})
        assert len(span.events) == 1
        assert span.events[0]["name"] == "checkpoint"
        assert span.events[0]["attributes"]["progress"] == 50

    def test_span_record_exception(self):
        """Test recording exceptions."""
        span = Span("test", "t1", "s1", None, time.time())
        try:
            raise ValueError("test error")
        except Exception as e:
            span.record_exception(e)

        assert span.status == "ERROR"
        assert "ValueError" in span.error
        assert len(span.events) == 1
        assert span.events[0]["name"] == "exception"

    def test_span_duration_ms(self):
        """Test duration calculation."""
        span = Span("test", "t1", "s1", None, time.time())
        time.sleep(0.01)
        span.end_time = time.time()
        assert span.duration_ms >= 10  # At least 10ms

    def test_span_duration_ms_none_when_not_ended(self):
        """Test duration is None when span not ended."""
        span = Span("test", "t1", "s1", None, time.time())
        assert span.duration_ms is None

    def test_span_get_span_context(self):
        """Test getting span context."""
        span = Span("test", "t1", "s1", "parent1", time.time())
        ctx = span.get_span_context()
        assert isinstance(ctx, SpanContext)
        assert ctx.trace_id == "t1"
        assert ctx.span_id == "s1"
        assert ctx.parent_span_id == "parent1"

    def test_span_to_dict(self):
        """Test conversion to dictionary."""
        span = Span("test.span", "t1", "s1", None, time.time())
        span.set_attribute("key", "value")
        span.end_time = time.time()

        d = span.to_dict()
        assert d["name"] == "test.span"
        assert d["trace_id"] == "t1"
        assert d["attributes"]["key"] == "value"
        assert d["status"] == "OK"


class TestSpanRecorder:
    """Test SpanRecorder class."""

    def test_record_span(self):
        """Test recording spans."""
        recorder = SpanRecorder(max_spans=100)
        span = Span("test", "t1", "s1", None, time.time())
        span.end_time = time.time()

        recorder.record(span)
        assert len(recorder.spans) == 1

    def test_get_spans_by_trace(self):
        """Test filtering by trace ID."""
        recorder = SpanRecorder()
        for i in range(5):
            span = Span(f"span{i}", "trace_a" if i < 3 else "trace_b", f"s{i}", None, time.time())
            span.end_time = time.time()
            recorder.record(span)

        trace_a = recorder.get_spans_by_trace("trace_a")
        assert len(trace_a) == 3

    def test_eviction_on_max_spans(self):
        """Test span eviction when max is reached."""
        recorder = SpanRecorder(max_spans=10)
        for i in range(15):
            span = Span(f"span{i}", "t1", f"s{i}", None, time.time())
            span.end_time = time.time()
            recorder.record(span)

        # Should have evicted half (5) when limit reached
        assert len(recorder.spans) <= 10

    def test_get_recent_spans(self):
        """Test getting recent spans."""
        recorder = SpanRecorder()
        for i in range(10):
            span = Span(f"span{i}", "t1", f"s{i}", None, time.time())
            span.end_time = time.time()
            recorder.record(span)

        recent = recorder.get_recent_spans(limit=5)
        assert len(recent) == 5

    def test_clear(self):
        """Test clearing recorder."""
        recorder = SpanRecorder()
        span = Span("test", "t1", "s1", None, time.time())
        span.end_time = time.time()
        recorder.record(span)

        recorder.clear()
        assert len(recorder.spans) == 0


class TestTracer:
    """Test Tracer class."""

    def test_span_context_manager(self):
        """Test span as context manager."""
        tracer = Tracer(log_spans=False)
        with tracer.span("test.operation", key="value") as span:
            span.set_attribute("result", "success")

        assert span.status == "OK"
        assert span.end_time is not None
        assert span.attributes["key"] == "value"
        assert span.attributes["result"] == "success"

    def test_span_exception_handling(self):
        """Test span exception recording."""
        tracer = Tracer(log_spans=False)
        with pytest.raises(ValueError):
            with tracer.span("failing.operation") as span:
                raise ValueError("test error")

        assert span.status == "ERROR"
        assert "ValueError" in span.error

    def test_nested_spans(self):
        """Test nested span hierarchy."""
        tracer = Tracer(log_spans=False)
        with tracer.span("parent") as parent:
            with tracer.span("child") as child:
                assert child.parent_span_id == parent.span_id
                assert child.trace_id == parent.trace_id

    def test_get_current_span(self):
        """Test getting current span."""
        tracer = Tracer(log_spans=False)
        assert tracer.get_current_span() is None

        with tracer.span("test") as span:
            assert tracer.get_current_span() == span

        assert tracer.get_current_span() is None

    def test_explicit_parent(self):
        """Test explicit parent span."""
        tracer = Tracer(log_spans=False)
        with tracer.span("parent") as parent:
            pass

        with tracer.span("child", parent=parent) as child:
            assert child.parent_span_id == parent.span_id

    def test_explicit_trace_id(self):
        """Test explicit trace ID."""
        tracer = Tracer(log_spans=False)
        with tracer.span("test", trace_id="custom_trace_123") as span:
            assert span.trace_id == "custom_trace_123"


class TestDebateContext:
    """Test debate context management."""

    def test_set_and_get_debate_context(self):
        """Test setting and getting debate context."""
        set_debate_context("debate_123", round=1)
        ctx = get_debate_context()
        assert ctx["debate_id"] == "debate_123"
        assert ctx["round"] == 1

    def test_get_debate_id(self):
        """Test getting debate ID from context."""
        set_debate_context("debate_456")
        assert get_debate_id() == "debate_456"


class TestDebateMetrics:
    """Test DebateMetrics class."""

    def test_metrics_creation(self):
        """Test metrics initialization."""
        metrics = DebateMetrics(debate_id="test_123")
        assert metrics.debate_id == "test_123"
        assert metrics.agent_calls == 0

    def test_record_agent_latency(self):
        """Test recording agent latency."""
        metrics = DebateMetrics(debate_id="test")
        metrics.record_agent_latency("agent1", 100.0)
        metrics.record_agent_latency("agent1", 150.0)
        metrics.record_agent_latency("agent2", 200.0)

        assert metrics.agent_calls == 3
        assert metrics.get_agent_avg_latency("agent1") == 125.0
        assert metrics.get_agent_avg_latency("agent2") == 200.0

    def test_metrics_to_dict(self):
        """Test metrics to dictionary."""
        metrics = DebateMetrics(debate_id="test")
        metrics.total_duration_ms = 5000
        metrics.rounds_completed = 3
        metrics.consensus_reached = True

        d = metrics.to_dict()
        assert d["debate_id"] == "test"
        assert d["total_duration_ms"] == 5000
        assert d["rounds_completed"] == 3
        assert d["consensus_reached"] is True

    def test_get_metrics_creates_new(self):
        """Test get_metrics creates new metrics if not exists."""
        clear_metrics()
        metrics = get_metrics("new_debate")
        assert metrics.debate_id == "new_debate"

    def test_get_metrics_returns_existing(self):
        """Test get_metrics returns existing metrics."""
        clear_metrics()
        metrics1 = get_metrics("debate_x")
        metrics1.agent_calls = 5

        metrics2 = get_metrics("debate_x")
        assert metrics2.agent_calls == 5

    def test_clear_metrics(self):
        """Test clearing specific metrics."""
        clear_metrics()
        get_metrics("debate_a")
        get_metrics("debate_b")

        clear_metrics("debate_a")
        assert get_metrics("debate_b").debate_id == "debate_b"


class TestGlobalTracer:
    """Test global tracer management."""

    def test_get_tracer_singleton(self):
        """Test get_tracer returns singleton."""
        tracer1 = get_tracer()
        tracer2 = get_tracer()
        assert tracer1 is tracer2

    def test_set_tracer(self):
        """Test setting custom tracer."""
        custom = Tracer(service_name="custom", log_spans=False)
        set_tracer(custom)
        assert get_tracer() is custom
