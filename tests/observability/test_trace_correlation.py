"""
Tests for aragora.observability.trace_correlation module.

Covers:
- TraceContext dataclass (trace_id_short, as_labels)
- should_sample_trace_id sampling
- get_traced_latency_samples / clear_traced_latency_samples
- get_slow_traces filtering
- generate_exemplar_line formatting
- _record_traced_latency bounded buffer
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.observability.trace_correlation import (
    TraceContext,
    clear_traced_latency_samples,
    generate_exemplar_line,
    get_slow_traces,
    get_traced_latency_samples,
    should_sample_trace_id,
    _record_traced_latency,
    _TRACED_LATENCY_MAX_SAMPLES,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _clear_samples():
    """Clear traced latency samples between tests."""
    clear_traced_latency_samples()
    yield
    clear_traced_latency_samples()


# =============================================================================
# TestTraceContext
# =============================================================================


class TestTraceContext:
    """Tests for TraceContext dataclass."""

    def test_defaults(self):
        """Should have None defaults."""
        ctx = TraceContext()
        assert ctx.trace_id is None
        assert ctx.span_id is None
        assert ctx.sampled is False

    def test_trace_id_short(self):
        """Should return first 8 chars."""
        ctx = TraceContext(trace_id="abcdef1234567890")
        assert ctx.trace_id_short == "abcdef12"

    def test_trace_id_short_none(self):
        """Should return None when no trace_id."""
        ctx = TraceContext()
        assert ctx.trace_id_short is None

    def test_as_labels_when_sampled(self):
        """Should include trace_id when sampled."""
        ctx = TraceContext(trace_id="abcdef1234567890", sampled=True)
        labels = ctx.as_labels()
        assert "trace_id" in labels
        assert labels["trace_id"] == "abcdef12"

    def test_as_labels_not_sampled(self):
        """Should return empty dict when not sampled."""
        ctx = TraceContext(trace_id="abcdef1234567890", sampled=False)
        labels = ctx.as_labels()
        assert labels == {}

    def test_as_labels_no_trace_id(self):
        """Should return empty dict when no trace_id."""
        ctx = TraceContext(sampled=True)
        labels = ctx.as_labels()
        assert labels == {}


# =============================================================================
# TestShouldSampleTraceId
# =============================================================================


class TestShouldSampleTraceId:
    """Tests for should_sample_trace_id."""

    def test_returns_bool(self):
        """Should return a boolean."""
        result = should_sample_trace_id()
        assert isinstance(result, bool)

    @patch("aragora.observability.trace_correlation.TRACE_METRIC_SAMPLE_RATE", 1.0)
    def test_always_samples_at_rate_1(self):
        """Should always sample at rate 1.0."""
        results = [should_sample_trace_id() for _ in range(100)]
        assert all(results)

    @patch("aragora.observability.trace_correlation.TRACE_METRIC_SAMPLE_RATE", 0.0)
    def test_never_samples_at_rate_0(self):
        """Should never sample at rate 0.0."""
        results = [should_sample_trace_id() for _ in range(100)]
        assert not any(results)


# =============================================================================
# TestTracedLatencySamples
# =============================================================================


class TestTracedLatencySamples:
    """Tests for traced latency sample buffer."""

    def test_empty_initially(self):
        """Should be empty initially."""
        assert get_traced_latency_samples() == []

    def test_record_sample(self):
        """Should record samples."""
        _record_traced_latency("/api/test", "GET", 0.5, "abc123")
        samples = get_traced_latency_samples()
        assert len(samples) == 1
        assert samples[0] == ("/api/test", "GET", 0.5, "abc123")

    def test_clear_samples(self):
        """Should clear all samples."""
        _record_traced_latency("/api/test", "GET", 0.5, "abc123")
        clear_traced_latency_samples()
        assert get_traced_latency_samples() == []

    def test_bounded_buffer(self):
        """Buffer should not grow beyond max."""
        for i in range(_TRACED_LATENCY_MAX_SAMPLES + 100):
            _record_traced_latency(f"/api/{i}", "GET", 0.1, f"trace{i}")

        samples = get_traced_latency_samples()
        assert len(samples) <= _TRACED_LATENCY_MAX_SAMPLES


# =============================================================================
# TestGetSlowTraces
# =============================================================================


class TestGetSlowTraces:
    """Tests for get_slow_traces."""

    def test_empty_when_no_samples(self):
        """Should return empty list."""
        assert get_slow_traces() == []

    def test_filters_by_threshold(self):
        """Should filter traces by threshold."""
        _record_traced_latency("/api/fast", "GET", 0.1, "trace1")
        _record_traced_latency("/api/slow", "GET", 2.0, "trace2")

        slow = get_slow_traces(threshold_seconds=1.0)
        assert len(slow) == 1
        assert slow[0]["endpoint"] == "/api/slow"
        assert slow[0]["trace_id"] == "trace2"

    def test_sorted_by_duration_descending(self):
        """Should sort by duration descending."""
        _record_traced_latency("/api/a", "GET", 1.5, "t1")
        _record_traced_latency("/api/b", "GET", 3.0, "t2")
        _record_traced_latency("/api/c", "GET", 2.0, "t3")

        slow = get_slow_traces(threshold_seconds=1.0)
        durations = [s["duration_seconds"] for s in slow]
        assert durations == sorted(durations, reverse=True)

    def test_result_dict_format(self):
        """Should return properly formatted dicts."""
        _record_traced_latency("/api/test", "POST", 1.5, "abc12345")

        slow = get_slow_traces(threshold_seconds=1.0)
        assert len(slow) == 1
        entry = slow[0]
        assert entry["endpoint"] == "/api/test"
        assert entry["method"] == "POST"
        assert entry["duration_seconds"] == 1.5
        assert entry["trace_id"] == "abc12345"


# =============================================================================
# TestGenerateExemplarLine
# =============================================================================


class TestGenerateExemplarLine:
    """Tests for generate_exemplar_line."""

    def test_format(self):
        """Should generate correct Prometheus exemplar format."""
        result = generate_exemplar_line("abc123", 0.5)
        assert 'trace_id="abc123"' in result
        assert "0.5" in result

    def test_includes_hash_prefix(self):
        """Should include # prefix for Prometheus."""
        result = generate_exemplar_line("trace1", 1.0)
        assert result.strip().startswith("#")
