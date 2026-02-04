"""Tests for automatic handler instrumentation.

Validates that ``instrumented_handler`` and ``auto_instrument_handler``
correctly wrap handler methods with tracing and metrics while
gracefully degrading when observability is unavailable.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handler_registry.instrumented import (
    _INSTRUMENTABLE_METHODS,
    _get_tracing,
    auto_instrument_handler,
    instrumented_handler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeHandlerResult:
    """Simulates a handler result with a ``status_code`` attribute."""

    status_code: int
    body: bytes = b""


class _StubHandler:
    """Minimal handler used in ``auto_instrument_handler`` tests."""

    def handle(self, path: str, query: dict, request_handler: Any) -> FakeHandlerResult:
        return FakeHandlerResult(status_code=200, body=b"ok")

    def handle_post(self, path: str, query: dict, request_handler: Any) -> FakeHandlerResult:
        return FakeHandlerResult(status_code=201, body=b"created")

    def can_handle(self, path: str) -> bool:
        return True


class _ErrorHandler:
    """Handler whose ``handle`` raises."""

    def handle(self, path: str, query: dict, request_handler: Any) -> Any:
        raise ValueError("boom")


class _TupleHandler:
    """Handler returning a plain tuple."""

    def handle(self, path: str, query: dict, request_handler: Any) -> tuple:
        return (202, {}, b"accepted")


class _NoneHandler:
    """Handler returning None."""

    def handle(self, path: str, query: dict, request_handler: Any) -> None:
        return None


class _ReadOnlyHandler:
    """Handler where setattr on methods is forbidden."""

    __slots__ = ()

    def handle(self, path: str, query: dict, request_handler: Any) -> FakeHandlerResult:
        return FakeHandlerResult(status_code=200)


# ---------------------------------------------------------------------------
# Tests for instrumented_handler decorator
# ---------------------------------------------------------------------------


class TestInstrumentedHandler:
    """Unit tests for the ``instrumented_handler`` decorator."""

    def test_wraps_function_and_preserves_return(self) -> None:
        """Decorated function should return the original result."""

        @instrumented_handler("TestHandler", "handle")
        def my_handler() -> FakeHandlerResult:
            return FakeHandlerResult(status_code=200)

        result = my_handler()
        assert result.status_code == 200

    def test_extracts_status_from_dataclass(self) -> None:
        """Status code should be extracted from a dataclass result."""

        @instrumented_handler("TestHandler", "handle")
        def my_handler() -> FakeHandlerResult:
            return FakeHandlerResult(status_code=404)

        result = my_handler()
        assert result.status_code == 404

    def test_extracts_status_from_tuple(self) -> None:
        """Status code should be extracted from a tuple result."""

        @instrumented_handler("TestHandler", "handle")
        def my_handler() -> tuple:
            return (201, {}, b"body")

        result = my_handler()
        assert result == (201, {}, b"body")

    def test_none_result_defaults_to_200(self) -> None:
        """A None result should default to status 200 internally."""

        @instrumented_handler("TestHandler", "handle")
        def my_handler() -> None:
            return None

        # Should not raise
        result = my_handler()
        assert result is None

    def test_non_standard_result_defaults_to_200(self) -> None:
        """An arbitrary non-tuple/non-dataclass result should default to 200."""

        @instrumented_handler("TestHandler", "handle")
        def my_handler() -> str:
            return "plain string"

        result = my_handler()
        assert result == "plain string"

    def test_exception_propagates(self) -> None:
        """Exceptions should propagate through the wrapper."""

        @instrumented_handler("TestHandler", "handle")
        def my_handler() -> None:
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError, match="fail"):
            my_handler()

    def test_preserves_function_name(self) -> None:
        """The wrapper should preserve the original function's __name__."""

        @instrumented_handler("TestHandler", "handle")
        def my_special_handler() -> None:
            pass

        assert my_special_handler.__name__ == "my_special_handler"

    def test_latency_is_recorded(self) -> None:
        """When observability is available, latency should be recorded."""
        mock_record = MagicMock()
        mock_span = MagicMock()

        with patch(
            "aragora.server.handler_registry.instrumented._get_tracing",
            return_value=(mock_span, mock_record),
        ):

            @instrumented_handler("TestHandler", "handle")
            def my_handler() -> FakeHandlerResult:
                return FakeHandlerResult(status_code=200)

            my_handler()

        mock_record.assert_called_once()
        call_args = mock_record.call_args
        assert call_args[0][0] == "HANDLE"  # method derived from "handle"
        assert call_args[0][1] == "TestHandler"
        assert call_args[0][2] == 200  # status
        assert call_args[0][3] >= 0  # duration

    def test_span_is_created(self) -> None:
        """When observability is available, a span should be started."""
        mock_start_span = MagicMock()
        mock_record = MagicMock()

        with patch(
            "aragora.server.handler_registry.instrumented._get_tracing",
            return_value=(mock_start_span, mock_record),
        ):

            @instrumented_handler("TestHandler", "handle_post")
            def my_handler() -> FakeHandlerResult:
                return FakeHandlerResult(status_code=201)

            my_handler()

        mock_start_span.assert_called_once()
        span_name = mock_start_span.call_args[0][0]
        assert span_name == "handler.TestHandler.handle_post"

    def test_handles_record_failure_gracefully(self) -> None:
        """If recording metrics raises, the handler should still return."""
        mock_record = MagicMock(side_effect=RuntimeError("metrics down"))
        mock_span = MagicMock(side_effect=RuntimeError("tracing down"))

        with patch(
            "aragora.server.handler_registry.instrumented._get_tracing",
            return_value=(mock_span, mock_record),
        ):

            @instrumented_handler("TestHandler", "handle")
            def my_handler() -> FakeHandlerResult:
                return FakeHandlerResult(status_code=200)

            result = my_handler()
            assert result.status_code == 200


# ---------------------------------------------------------------------------
# Tests for graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Ensure the module works even when observability is not installed."""

    def test_get_tracing_returns_none_when_unavailable(self) -> None:
        """_get_tracing should return (None, None) when imports fail."""
        import aragora.server.handler_registry.instrumented as mod

        # Reset the cached state
        original = mod._tracing_available
        mod._tracing_available = None

        with patch.dict("sys.modules", {"aragora.observability.handler_instrumentation": None}):
            mod._tracing_available = None
            result = mod._get_tracing()
            assert result == (None, None)
            assert mod._tracing_available is False

        # Restore
        mod._tracing_available = original

    def test_decorator_works_without_observability(self) -> None:
        """Decorated handler should work when _get_tracing returns (None, None)."""
        with patch(
            "aragora.server.handler_registry.instrumented._get_tracing",
            return_value=(None, None),
        ):

            @instrumented_handler("TestHandler", "handle")
            def my_handler() -> FakeHandlerResult:
                return FakeHandlerResult(status_code=200)

            result = my_handler()
            assert result.status_code == 200


# ---------------------------------------------------------------------------
# Tests for auto_instrument_handler
# ---------------------------------------------------------------------------


class TestAutoInstrumentHandler:
    """Tests for class-level auto-instrumentation."""

    def test_instruments_handle_method(self) -> None:
        """handle() should be wrapped after auto_instrument_handler."""
        handler = _StubHandler()
        original_handle = handler.handle
        auto_instrument_handler(handler)

        # The method should have been replaced
        assert handler.handle is not original_handle
        # But it should still work
        result = handler.handle("/test", {}, None)
        assert result.status_code == 200

    def test_instruments_handle_post(self) -> None:
        """handle_post() should be wrapped."""
        handler = _StubHandler()
        original = handler.handle_post
        auto_instrument_handler(handler)

        assert handler.handle_post is not original
        result = handler.handle_post("/test", {}, None)
        assert result.status_code == 201

    def test_skips_missing_methods(self) -> None:
        """Methods not present on the handler should be silently skipped."""

        class MinimalHandler:
            def handle(self, path: str, query: dict, rh: Any) -> None:
                return None

        handler = MinimalHandler()
        # Should not raise
        auto_instrument_handler(handler)
        assert handler.handle("/x", {}, None) is None

    def test_exception_still_propagates(self) -> None:
        """Exceptions from the original handler should propagate through."""
        handler = _ErrorHandler()
        auto_instrument_handler(handler)

        with pytest.raises(ValueError, match="boom"):
            handler.handle("/test", {}, None)

    def test_tuple_result_passthrough(self) -> None:
        """Tuple results should be returned unmodified."""
        handler = _TupleHandler()
        auto_instrument_handler(handler)

        result = handler.handle("/test", {}, None)
        assert result == (202, {}, b"accepted")

    def test_none_result_passthrough(self) -> None:
        """None results should be returned unmodified."""
        handler = _NoneHandler()
        auto_instrument_handler(handler)

        result = handler.handle("/test", {}, None)
        assert result is None

    def test_returns_same_instance(self) -> None:
        """auto_instrument_handler should return the same instance."""
        handler = _StubHandler()
        returned = auto_instrument_handler(handler)
        assert returned is handler

    def test_readonly_handler_does_not_crash(self) -> None:
        """Handlers that forbid setattr should not crash instrumentation."""
        handler = _ReadOnlyHandler()
        # __slots__ prevents setting arbitrary attrs but bound methods
        # may still work. The important thing is no exception is raised.
        auto_instrument_handler(handler)
