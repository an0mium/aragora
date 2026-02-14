"""
Tests for aragora.server.middleware.correlation - Request correlation middleware.

Tests cover:
- CorrelationContext dataclass (as_headers, as_log_dict, extras)
- init_correlation from scratch (generates IDs)
- init_correlation from headers (propagates existing IDs)
- init_correlation with explicit overrides
- Traceparent header parsing
- get_correlation context variable behaviour
- get_or_create_correlation lazy creation
- CorrelationLogFilter log record injection
- correlation_log_extra convenience function
- Nested / sequential request handling (context isolation)
- Legacy context variable back-population
- Module __all__ exports
"""

from __future__ import annotations

import logging
from contextvars import copy_context
from unittest.mock import patch

import pytest


# ===========================================================================
# CorrelationContext dataclass
# ===========================================================================


class TestCorrelationContext:
    """Tests for the CorrelationContext dataclass."""

    def test_as_headers_basic(self):
        """as_headers returns propagation headers with W3C traceparent."""
        from aragora.server.middleware.correlation import CorrelationContext

        ctx = CorrelationContext(
            request_id="req-abc123",
            trace_id="a" * 32,
            span_id="b" * 16,
        )
        headers = ctx.as_headers()

        assert headers["X-Request-ID"] == "req-abc123"
        assert headers["X-Trace-ID"] == "a" * 32
        assert headers["X-Span-ID"] == "b" * 16
        assert headers["traceparent"] == f"00-{'a' * 32}-{'b' * 16}-01"
        # No parent span header when parent_span_id is None
        assert "X-Parent-Span-ID" not in headers

    def test_as_headers_with_parent_span(self):
        """as_headers includes parent span header when set."""
        from aragora.server.middleware.correlation import CorrelationContext

        ctx = CorrelationContext(
            request_id="req-abc123",
            trace_id="a" * 32,
            span_id="b" * 16,
            parent_span_id="c" * 16,
        )
        headers = ctx.as_headers()
        assert headers["X-Parent-Span-ID"] == "c" * 16

    def test_as_headers_pads_short_ids(self):
        """traceparent pads short trace/span IDs to standard widths."""
        from aragora.server.middleware.correlation import CorrelationContext

        ctx = CorrelationContext(
            request_id="req-short",
            trace_id="abc",
            span_id="def",
        )
        headers = ctx.as_headers()
        tp = headers["traceparent"]
        parts = tp.split("-")
        assert parts[0] == "00"
        assert len(parts[1]) == 32  # padded to 32 chars
        assert len(parts[2]) == 16  # padded to 16 chars
        assert parts[3] == "01"

    def test_as_log_dict_basic(self):
        """as_log_dict returns flat dict with core IDs."""
        from aragora.server.middleware.correlation import CorrelationContext

        ctx = CorrelationContext(
            request_id="req-log",
            trace_id="tid123",
            span_id="sid456",
        )
        d = ctx.as_log_dict()
        assert d["request_id"] == "req-log"
        assert d["trace_id"] == "tid123"
        assert d["span_id"] == "sid456"
        assert "parent_span_id" not in d

    def test_as_log_dict_with_parent_and_extras(self):
        """as_log_dict includes parent_span_id and extras when present."""
        from aragora.server.middleware.correlation import CorrelationContext

        ctx = CorrelationContext(
            request_id="req-x",
            trace_id="tid",
            span_id="sid",
            parent_span_id="psid",
            extras={"debate_id": "d-42", "tenant": "acme"},
        )
        d = ctx.as_log_dict()
        assert d["parent_span_id"] == "psid"
        assert d["debate_id"] == "d-42"
        assert d["tenant"] == "acme"

    def test_extras_default_empty(self):
        """extras defaults to an empty dict."""
        from aragora.server.middleware.correlation import CorrelationContext

        ctx = CorrelationContext(request_id="r", trace_id="t", span_id="s")
        assert ctx.extras == {}


# ===========================================================================
# init_correlation
# ===========================================================================


class TestInitCorrelation:
    """Tests for the init_correlation function."""

    def test_generates_ids_when_no_headers(self):
        """Calling with no args generates fresh IDs."""
        from aragora.server.middleware.correlation import init_correlation

        ctx = init_correlation()
        assert ctx.request_id.startswith("req-")
        assert len(ctx.trace_id) == 32
        assert len(ctx.span_id) == 16
        assert ctx.parent_span_id is None

    def test_propagates_request_id_from_header(self):
        """Request ID is extracted from the X-Request-ID header."""
        from aragora.server.middleware.correlation import init_correlation

        ctx = init_correlation(headers={"X-Request-ID": "req-from-upstream"})
        assert ctx.request_id == "req-from-upstream"

    def test_propagates_request_id_lowercase_header(self):
        """Request ID is extracted even with lowercase header name."""
        from aragora.server.middleware.correlation import init_correlation

        ctx = init_correlation(headers={"x-request-id": "req-lower"})
        assert ctx.request_id == "req-lower"

    def test_propagates_trace_id_from_header(self):
        """Trace ID is extracted from X-Trace-ID header."""
        from aragora.server.middleware.correlation import init_correlation

        tid = "a" * 32
        ctx = init_correlation(headers={"X-Trace-ID": tid})
        assert ctx.trace_id == tid

    def test_propagates_trace_id_lowercase_header(self):
        """Trace ID is extracted with lowercase header name."""
        from aragora.server.middleware.correlation import init_correlation

        tid = "b" * 32
        ctx = init_correlation(headers={"x-trace-id": tid})
        assert ctx.trace_id == tid

    def test_extracts_trace_id_from_traceparent(self):
        """Trace ID and parent span are parsed from W3C traceparent header."""
        from aragora.server.middleware.correlation import init_correlation

        tid = "f" * 32
        parent = "e" * 16
        traceparent = f"00-{tid}-{parent}-01"
        ctx = init_correlation(headers={"traceparent": traceparent})
        assert ctx.trace_id == tid
        assert ctx.parent_span_id == parent

    def test_traceparent_with_only_version_and_trace(self):
        """traceparent with only 2 parts still extracts trace_id."""
        from aragora.server.middleware.correlation import init_correlation

        tid = "d" * 32
        traceparent = f"00-{tid}"
        ctx = init_correlation(headers={"traceparent": traceparent})
        assert ctx.trace_id == tid
        assert ctx.parent_span_id is None

    def test_trace_id_header_takes_precedence_over_traceparent(self):
        """Explicit X-Trace-ID header takes precedence over traceparent."""
        from aragora.server.middleware.correlation import init_correlation

        explicit_tid = "1" * 32
        traceparent_tid = "2" * 32
        ctx = init_correlation(
            headers={
                "X-Trace-ID": explicit_tid,
                "traceparent": f"00-{traceparent_tid}-{'0' * 16}-01",
            }
        )
        assert ctx.trace_id == explicit_tid

    def test_explicit_overrides_take_precedence(self):
        """Keyword argument overrides beat header values."""
        from aragora.server.middleware.correlation import init_correlation

        ctx = init_correlation(
            headers={"X-Request-ID": "from-header"},
            request_id="explicit-rid",
            trace_id="explicit-tid",
            span_id="explicit-sid",
            parent_span_id="explicit-psid",
        )
        assert ctx.request_id == "explicit-rid"
        assert ctx.trace_id == "explicit-tid"
        assert ctx.span_id == "explicit-sid"
        assert ctx.parent_span_id == "explicit-psid"

    def test_propagates_parent_span_from_header(self):
        """Parent span ID is extracted from X-Parent-Span-ID header."""
        from aragora.server.middleware.correlation import init_correlation

        psid = "c" * 16
        ctx = init_correlation(headers={"X-Parent-Span-ID": psid})
        assert ctx.parent_span_id == psid

    def test_parent_span_from_traceparent_not_overwritten_by_header(self):
        """When traceparent supplies parent span, explicit parent_span_id overrides."""
        from aragora.server.middleware.correlation import init_correlation

        ctx = init_correlation(
            headers={"traceparent": f"00-{'a' * 32}-{'b' * 16}-01"},
            parent_span_id="override",
        )
        # Explicit keyword override wins
        assert ctx.parent_span_id == "override"


# ===========================================================================
# Context variable management
# ===========================================================================


class TestContextVariables:
    """Tests for get_correlation and get_or_create_correlation."""

    def test_get_correlation_returns_none_without_init(self):
        """get_correlation returns None in a fresh context."""
        from aragora.server.middleware.correlation import (
            _correlation,
            get_correlation,
        )

        # Run in a copied context but reset the context var to simulate fresh state
        def _check():
            _correlation.set(None)
            return get_correlation()

        result = copy_context().run(_check)
        assert result is None

    def test_get_correlation_returns_set_context(self):
        """get_correlation returns the context set by init_correlation."""
        from aragora.server.middleware.correlation import (
            get_correlation,
            init_correlation,
        )

        def _check():
            ctx = init_correlation(request_id="req-get-test")
            fetched = get_correlation()
            return ctx, fetched

        ctx, fetched = copy_context().run(_check)
        assert fetched is ctx
        assert fetched.request_id == "req-get-test"

    def test_get_or_create_creates_when_missing(self):
        """get_or_create_correlation creates a new context if none exists."""
        from aragora.server.middleware.correlation import (
            _correlation,
            get_or_create_correlation,
        )

        def _check():
            # Ensure no correlation exists in this context copy
            _correlation.set(None)
            ctx = get_or_create_correlation()
            assert ctx is not None
            assert ctx.request_id.startswith("req-")
            return ctx

        copy_context().run(_check)

    def test_get_or_create_returns_existing(self):
        """get_or_create_correlation returns existing context if present."""
        from aragora.server.middleware.correlation import (
            get_or_create_correlation,
            init_correlation,
        )

        def _check():
            original = init_correlation(request_id="req-existing")
            fetched = get_or_create_correlation()
            return original, fetched

        original, fetched = copy_context().run(_check)
        assert fetched is original

    def test_nested_contexts_isolated(self):
        """Separate contextvars copies maintain independent correlation state."""
        from aragora.server.middleware.correlation import (
            get_correlation,
            init_correlation,
        )

        outer_ctx = None
        inner_ctx = None

        def _outer():
            nonlocal outer_ctx, inner_ctx
            outer_ctx = init_correlation(request_id="req-outer")

            # Inner context inherits but can override
            def _inner():
                nonlocal inner_ctx
                inner_ctx = init_correlation(request_id="req-inner")

            copy_context().run(_inner)
            # Outer should still see its own context
            return get_correlation()

        result = copy_context().run(_outer)
        assert result.request_id == "req-outer"
        assert inner_ctx.request_id == "req-inner"


# ===========================================================================
# Legacy context variable back-population
# ===========================================================================


class TestLegacyBackPopulation:
    """Tests that init_correlation back-populates legacy context vars."""

    def test_sets_legacy_request_id(self):
        """init_correlation calls set_current_request_id."""
        from aragora.server.middleware.correlation import init_correlation

        with patch("aragora.server.middleware.correlation.set_current_request_id") as mock_set:
            init_correlation(request_id="req-legacy")
            mock_set.assert_called_once_with("req-legacy")

    def test_sets_legacy_trace_id(self):
        """init_correlation calls set_trace_id."""
        from aragora.server.middleware.correlation import init_correlation

        with patch("aragora.server.middleware.correlation.set_trace_id") as mock_set:
            ctx = init_correlation(trace_id="tid-legacy")
            mock_set.assert_called_once_with("tid-legacy")

    def test_sets_legacy_span_id(self):
        """init_correlation calls set_span_id."""
        from aragora.server.middleware.correlation import init_correlation

        with patch("aragora.server.middleware.correlation.set_span_id") as mock_set:
            init_correlation(span_id="sid-legacy")
            mock_set.assert_called_once_with("sid-legacy")


# ===========================================================================
# CorrelationLogFilter
# ===========================================================================


class TestCorrelationLogFilter:
    """Tests for the CorrelationLogFilter logging filter."""

    def _make_record(self) -> logging.LogRecord:
        """Create a minimal LogRecord for testing."""
        return logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )

    def test_injects_ids_when_context_set(self):
        """Filter injects correlation IDs from the active context."""
        from aragora.server.middleware.correlation import (
            CorrelationLogFilter,
            init_correlation,
        )

        def _check():
            init_correlation(
                request_id="req-filter",
                trace_id="tid-filter",
                span_id="sid-filter",
            )
            filt = CorrelationLogFilter()
            record = self._make_record()
            result = filt.filter(record)
            return result, record

        result, record = copy_context().run(_check)
        assert result is True
        assert record.request_id == "req-filter"  # type: ignore[attr-defined]
        assert record.trace_id == "tid-filter"  # type: ignore[attr-defined]
        assert record.span_id == "sid-filter"  # type: ignore[attr-defined]

    def test_sets_empty_strings_when_no_context(self):
        """Filter sets empty strings when no correlation context exists."""
        from aragora.server.middleware.correlation import (
            CorrelationLogFilter,
            _correlation,
        )

        def _check():
            _correlation.set(None)
            filt = CorrelationLogFilter()
            record = self._make_record()
            result = filt.filter(record)
            return result, record

        result, record = copy_context().run(_check)
        assert result is True
        assert record.request_id == ""  # type: ignore[attr-defined]
        assert record.trace_id == ""  # type: ignore[attr-defined]
        assert record.span_id == ""  # type: ignore[attr-defined]

    def test_filter_always_returns_true(self):
        """Filter should always return True (never suppress records)."""
        from aragora.server.middleware.correlation import CorrelationLogFilter

        filt = CorrelationLogFilter()

        def _check():
            record = self._make_record()
            return filt.filter(record)

        result = copy_context().run(_check)
        assert result is True


# ===========================================================================
# correlation_log_extra
# ===========================================================================


class TestCorrelationLogExtra:
    """Tests for the correlation_log_extra convenience function."""

    def test_returns_log_dict_when_context_set(self):
        """Returns the as_log_dict from active context."""
        from aragora.server.middleware.correlation import (
            correlation_log_extra,
            init_correlation,
        )

        def _check():
            ctx = init_correlation(
                request_id="req-extra",
                trace_id="tid-extra",
                span_id="sid-extra",
            )
            ctx.extras["env"] = "test"
            return correlation_log_extra()

        extra = copy_context().run(_check)
        assert extra["request_id"] == "req-extra"
        assert extra["trace_id"] == "tid-extra"
        assert extra["span_id"] == "sid-extra"
        assert extra["env"] == "test"

    def test_returns_empty_dict_when_no_context(self):
        """Returns empty dict when no correlation context exists."""
        from aragora.server.middleware.correlation import (
            _correlation,
            correlation_log_extra,
        )

        def _check():
            _correlation.set(None)
            return correlation_log_extra()

        result = copy_context().run(_check)
        assert result == {}


# ===========================================================================
# Module exports
# ===========================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports(self):
        """__all__ contains all public API names."""
        from aragora.server.middleware import correlation

        expected = {
            "CorrelationContext",
            "CorrelationLogFilter",
            "correlation_log_extra",
            "get_correlation",
            "get_or_create_correlation",
            "init_correlation",
        }
        assert set(correlation.__all__) == expected

    def test_all_exports_are_importable(self):
        """Every name in __all__ is actually defined in the module."""
        from aragora.server.middleware import correlation

        for name in correlation.__all__:
            assert hasattr(correlation, name), f"{name} not found in module"
