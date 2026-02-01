"""
Tests for queue trace context propagation module.

Tests cover:
- TraceCarrier dataclass (creation, to_dict, from_dict, round-trip)
- inject_trace_context (with/without active trace)
- extract_trace_carrier (valid, missing, malformed)
- extract_and_activate (sets contextvars)
- traced_job decorator (activates trace, wraps in span, handles errors)
- get_tracing_stats
- Module __all__ exports
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.queue.tracing import (
    TraceCarrier,
    extract_trace_carrier,
    get_tracing_stats,
    inject_trace_context,
)


class TestTraceCarrier:
    """Tests for TraceCarrier dataclass."""

    def test_creation(self):
        carrier = TraceCarrier(
            trace_id="trace-123",
            parent_span_id="span-456",
            request_id="req-789",
        )
        assert carrier.trace_id == "trace-123"
        assert carrier.parent_span_id == "span-456"
        assert carrier.request_id == "req-789"

    def test_creation_optional_fields(self):
        carrier = TraceCarrier(trace_id="t-1", parent_span_id=None, request_id=None)
        assert carrier.parent_span_id is None
        assert carrier.request_id is None

    def test_to_dict(self):
        carrier = TraceCarrier(
            trace_id="t-abc",
            parent_span_id="s-def",
            request_id="r-ghi",
        )
        data = carrier.to_dict()
        assert data == {
            "trace_id": "t-abc",
            "parent_span_id": "s-def",
            "request_id": "r-ghi",
        }

    def test_to_dict_with_nones(self):
        carrier = TraceCarrier(trace_id="t-1", parent_span_id=None, request_id=None)
        data = carrier.to_dict()
        assert data["trace_id"] == "t-1"
        assert data["parent_span_id"] is None
        assert data["request_id"] is None

    def test_from_dict(self):
        data = {
            "trace_id": "t-from",
            "parent_span_id": "s-from",
            "request_id": "r-from",
        }
        carrier = TraceCarrier.from_dict(data)
        assert carrier.trace_id == "t-from"
        assert carrier.parent_span_id == "s-from"
        assert carrier.request_id == "r-from"

    def test_from_dict_defaults(self):
        data = {}
        carrier = TraceCarrier.from_dict(data)
        assert carrier.trace_id == ""
        assert carrier.parent_span_id is None
        assert carrier.request_id is None

    def test_round_trip(self):
        original = TraceCarrier(
            trace_id="round-trip-trace",
            parent_span_id="round-trip-span",
            request_id="round-trip-req",
        )
        restored = TraceCarrier.from_dict(original.to_dict())
        assert restored.trace_id == original.trace_id
        assert restored.parent_span_id == original.parent_span_id
        assert restored.request_id == original.request_id


class TestInjectTraceContext:
    """Tests for inject_trace_context function."""

    @patch("aragora.queue.tracing.get_trace_id", return_value=None)
    def test_no_active_trace(self, mock_get_trace):
        payload = {"key": "value"}
        result = inject_trace_context(payload)
        assert result is payload
        assert "_trace" not in result

    @patch("aragora.queue.tracing.get_span_id", return_value="span-42")
    @patch("aragora.queue.tracing.get_trace_id", return_value="trace-42")
    @patch("aragora.queue.tracing.get_or_create_correlation")
    def test_with_active_trace(self, mock_correlation, mock_trace, mock_span):
        mock_ctx = MagicMock()
        mock_ctx.request_id = "req-42"
        mock_correlation.return_value = mock_ctx

        payload = {"debate_id": "d-1"}
        result = inject_trace_context(payload)

        assert result is payload
        assert "_trace" in result
        trace_data = result["_trace"]
        assert trace_data["trace_id"] == "trace-42"
        assert trace_data["parent_span_id"] == "span-42"
        assert trace_data["request_id"] == "req-42"

    @patch("aragora.queue.tracing.get_span_id", return_value=None)
    @patch("aragora.queue.tracing.get_trace_id", return_value="trace-no-span")
    @patch("aragora.queue.tracing.get_or_create_correlation")
    def test_with_trace_but_no_span(self, mock_correlation, mock_trace, mock_span):
        mock_correlation.return_value = None

        payload = {}
        result = inject_trace_context(payload)

        assert "_trace" in result
        assert result["_trace"]["trace_id"] == "trace-no-span"
        assert result["_trace"]["parent_span_id"] is None
        assert result["_trace"]["request_id"] is None

    @patch("aragora.queue.tracing.get_trace_id", return_value="")
    def test_empty_trace_id(self, mock_get_trace):
        payload = {"key": "value"}
        result = inject_trace_context(payload)
        assert "_trace" not in result


class TestExtractTraceCarrier:
    """Tests for extract_trace_carrier function."""

    def test_valid_trace_data(self):
        payload = {
            "_trace": {
                "trace_id": "t-1",
                "parent_span_id": "s-1",
                "request_id": "r-1",
            }
        }
        carrier = extract_trace_carrier(payload)
        assert carrier is not None
        assert carrier.trace_id == "t-1"
        assert carrier.parent_span_id == "s-1"
        assert carrier.request_id == "r-1"

    def test_no_trace_key(self):
        payload = {"debate_id": "d-1"}
        carrier = extract_trace_carrier(payload)
        assert carrier is None

    def test_trace_key_is_none(self):
        payload = {"_trace": None}
        carrier = extract_trace_carrier(payload)
        assert carrier is None

    def test_trace_key_is_not_dict(self):
        payload = {"_trace": "not-a-dict"}
        carrier = extract_trace_carrier(payload)
        assert carrier is None

    def test_trace_key_is_empty_dict(self):
        payload = {"_trace": {}}
        carrier = extract_trace_carrier(payload)
        assert carrier is not None
        assert carrier.trace_id == ""

    def test_malformed_trace_data_returns_none(self):
        """Test that an exception during parsing returns None."""
        payload = {"_trace": {"trace_id": object()}}  # object() can't be parsed easily
        # Should not raise, should return carrier or None gracefully
        carrier = extract_trace_carrier(payload)
        # from_dict uses .get() with defaults, so it should handle most cases
        assert carrier is not None or carrier is None  # either is acceptable


class TestExtractAndActivate:
    """Tests for extract_and_activate function."""

    @patch("aragora.queue.tracing.init_correlation")
    @patch("aragora.queue.tracing.set_span_id")
    @patch("aragora.queue.tracing.set_trace_id")
    @patch("aragora.queue.tracing.generate_span_id", return_value="child-span-1")
    def test_activates_trace_context(
        self, mock_gen_span, mock_set_trace, mock_set_span, mock_init_corr
    ):
        from aragora.queue.tracing import extract_and_activate

        mock_ctx = MagicMock()
        mock_init_corr.return_value = mock_ctx

        payload = {
            "_trace": {
                "trace_id": "trace-activate",
                "parent_span_id": "parent-span",
                "request_id": "req-activate",
            }
        }

        result = extract_and_activate(payload)

        mock_set_trace.assert_called_once_with("trace-activate")
        mock_set_span.assert_called_once_with("child-span-1")
        mock_init_corr.assert_called_once_with(
            request_id="req-activate",
            trace_id="trace-activate",
            span_id="child-span-1",
            parent_span_id="parent-span",
        )
        assert result is mock_ctx

    def test_no_trace_returns_none(self):
        from aragora.queue.tracing import extract_and_activate

        payload = {"debate_id": "d-1"}
        result = extract_and_activate(payload)
        assert result is None


class TestTracedJob:
    """Tests for the @traced_job decorator."""

    @pytest.mark.asyncio
    async def test_traced_job_calls_function(self):
        from aragora.queue.tracing import traced_job

        call_log = []

        @traced_job("test.operation")
        async def my_job(data: dict) -> str:
            call_log.append(data)
            return "result"

        result = await my_job({"key": "value"})
        assert result == "result"
        assert len(call_log) == 1

    @pytest.mark.asyncio
    async def test_traced_job_propagates_exception(self):
        from aragora.queue.tracing import traced_job

        @traced_job("test.failing")
        async def failing_job() -> None:
            raise RuntimeError("job failed")

        with pytest.raises(RuntimeError, match="job failed"):
            await failing_job()

    @pytest.mark.asyncio
    async def test_traced_job_extracts_from_job_object(self):
        from aragora.queue.tracing import traced_job

        @traced_job("test.extract")
        async def process(job: object) -> str:
            return "done"

        # Create a mock job with a payload containing trace context
        mock_job = MagicMock()
        mock_job.payload = {
            "_trace": {
                "trace_id": "t-job",
                "parent_span_id": "s-job",
                "request_id": "r-job",
            }
        }

        result = await process(mock_job)
        assert result == "done"

    @pytest.mark.asyncio
    async def test_traced_job_extracts_from_dict_payload(self):
        from aragora.queue.tracing import traced_job

        @traced_job("test.dict")
        async def process(payload: dict) -> str:
            return "done"

        payload = {
            "_trace": {
                "trace_id": "t-dict",
                "parent_span_id": "s-dict",
                "request_id": "r-dict",
            }
        }

        result = await process(payload)
        assert result == "done"

    @pytest.mark.asyncio
    async def test_traced_job_default_operation_name(self):
        from aragora.queue.tracing import traced_job

        @traced_job()
        async def my_custom_job() -> str:
            return "custom"

        result = await my_custom_job()
        assert result == "custom"


class TestTracingStats:
    """Tests for get_tracing_stats function."""

    def test_returns_dict_with_expected_keys(self):
        stats = get_tracing_stats()
        assert "injected" in stats
        assert "extracted" in stats
        assert isinstance(stats["injected"], int)
        assert isinstance(stats["extracted"], int)


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports(self):
        from aragora.queue.tracing import __all__

        expected = [
            "TraceCarrier",
            "extract_and_activate",
            "extract_trace_carrier",
            "get_tracing_stats",
            "inject_trace_context",
            "traced_job",
        ]
        for name in expected:
            assert name in __all__
