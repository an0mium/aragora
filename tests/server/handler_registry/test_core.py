"""Tests for handler_registry core infrastructure."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handler_registry.core import (
    HandlerValidationError,
    RouteIndex,
    _safe_import,
    get_route_index,
    validate_all_handlers,
    validate_handler_class,
    validate_handler_instance,
    validate_handlers_on_init,
)


# ---------------------------------------------------------------------------
# Fake handlers for testing
# ---------------------------------------------------------------------------


class _GoodHandler:
    """A valid handler with all required attributes."""

    ROUTES = ["/api/test", "/api/test/detail"]
    ROUTE_PREFIXES = ["/api/extra/"]

    def __init__(self, ctx: dict | None = None) -> None:
        pass

    def can_handle(self, path: str) -> bool:
        return path.startswith("/api/test") or path.startswith("/api/extra/")

    def handle(self, path: str, query: dict, request_handler: Any) -> Any:
        return MagicMock(status_code=200, content_type="application/json", body=b"{}", headers={})


class _MinimalHandler:
    """Handler with can_handle and handle but no ROUTES."""

    def __init__(self, ctx: dict | None = None) -> None:
        pass

    def can_handle(self, path: str) -> bool:
        return path.startswith("/api/minimal")

    def handle(self, path: str, query: dict, request_handler: Any) -> Any:
        return None


class _BrokenCanHandle:
    """Handler whose can_handle raises."""

    def can_handle(self, path: str) -> bool:
        raise RuntimeError("broken handler")

    def handle(self, path: str, query: dict, request_handler: Any) -> Any:
        return None


class _NoCanHandle:
    """Handler missing can_handle."""

    def handle(self, path: str, query: dict, request_handler: Any) -> Any:
        return None


class _NoHandle:
    """Handler missing handle method."""

    def can_handle(self, path: str) -> bool:
        return False


class _NonCallableMethods:
    """Handler with non-callable can_handle and handle."""

    can_handle = "not callable"
    handle = 42


class _WrongReturnType:
    """Handler whose can_handle returns non-bool."""

    def can_handle(self, path: str) -> str:
        return "yes"  # type: ignore[return-value]

    def handle(self, path: str, query: dict, request_handler: Any) -> Any:
        return None


# ---------------------------------------------------------------------------
# _safe_import tests
# ---------------------------------------------------------------------------


class TestSafeImport:
    """Tests for safe handler import utility."""

    def test_import_existing_module(self) -> None:
        result = _safe_import("aragora.server.handler_registry.core", "RouteIndex")
        assert result is RouteIndex

    def test_import_nonexistent_module(self) -> None:
        result = _safe_import("aragora.nonexistent.module", "Foo")
        assert result is None

    def test_import_nonexistent_class(self) -> None:
        result = _safe_import("aragora.server.handler_registry.core", "NonexistentClass")
        assert result is None

    def test_import_from_stdlib(self) -> None:
        result = _safe_import("json", "JSONDecoder")
        import json

        assert result is json.JSONDecoder


# ---------------------------------------------------------------------------
# RouteIndex tests
# ---------------------------------------------------------------------------


class TestRouteIndex:
    """Tests for O(1) route lookup index."""

    def _build_index(self, handlers: list[tuple[str, Any]] | None = None) -> RouteIndex:
        """Build an index with test handlers."""
        idx = RouteIndex()
        if handlers is None:
            handlers = [
                ("_good_handler", _GoodHandler),
                ("_minimal_handler", _MinimalHandler),
            ]
        mixin = MagicMock()
        for attr_name, handler_class in handlers:
            setattr(mixin, attr_name, handler_class())
        idx.build(mixin, handlers)
        return idx

    def test_init_empty(self) -> None:
        idx = RouteIndex()
        assert idx._exact_routes == {}
        assert idx._prefix_routes == []

    def test_build_populates_exact_routes(self) -> None:
        idx = self._build_index()
        # _GoodHandler has ROUTES = ["/api/test", "/api/test/detail"]
        assert "/api/test" in idx._exact_routes
        assert "/api/test/detail" in idx._exact_routes

    def test_build_populates_prefix_routes(self) -> None:
        idx = self._build_index()
        prefixes = [p for p, _, _ in idx._prefix_routes]
        # _GoodHandler has ROUTE_PREFIXES = ["/api/extra/"]
        assert "/api/extra/" in prefixes

    def test_build_skips_none_handlers(self) -> None:
        mixin = MagicMock()
        mixin._good_handler = None
        idx = RouteIndex()
        idx.build(mixin, [("_good_handler", _GoodHandler)])
        assert len(idx._exact_routes) == 0

    def test_get_handler_exact_match(self) -> None:
        idx = self._build_index()
        with patch("aragora.server.versioning.strip_version_prefix", side_effect=lambda p: p):
            result = idx.get_handler("/api/test")
        assert result is not None
        assert result[0] == "_good_handler"

    def test_get_handler_version_stripped(self) -> None:
        idx = self._build_index()
        with patch(
            "aragora.server.versioning.strip_version_prefix",
            return_value="/api/test",
        ):
            result = idx.get_handler("/api/v1/test")
        assert result is not None
        assert result[0] == "_good_handler"

    def test_get_handler_prefix_match(self) -> None:
        idx = self._build_index()
        with patch("aragora.server.versioning.strip_version_prefix", side_effect=lambda p: p):
            result = idx.get_handler("/api/extra/something")
        assert result is not None
        assert result[0] == "_good_handler"

    def test_get_handler_no_match(self) -> None:
        idx = self._build_index()
        with patch("aragora.server.versioning.strip_version_prefix", side_effect=lambda p: p):
            result = idx.get_handler("/api/unknown/path")
        assert result is None

    def test_build_clears_previous_index(self) -> None:
        idx = self._build_index()
        count_before = len(idx._exact_routes)
        assert count_before > 0
        # Rebuild with empty registry
        mixin = MagicMock()
        idx.build(mixin, [])
        assert len(idx._exact_routes) == 0

    def test_duplicate_exact_route_first_wins(self) -> None:
        """First handler to register an exact route wins."""
        idx = RouteIndex()
        mixin = MagicMock()

        h1 = _GoodHandler()
        h2 = _GoodHandler()
        mixin._handler1 = h1
        mixin._handler2 = h2

        idx.build(mixin, [("_handler1", _GoodHandler), ("_handler2", _GoodHandler)])
        with patch("aragora.server.versioning.strip_version_prefix", side_effect=lambda p: p):
            result = idx.get_handler("/api/test")
        assert result is not None
        assert result[0] == "_handler1"

    def test_handler_prefixes_from_build_patterns(self) -> None:
        """PREFIX_PATTERNS in build() should register prefix routes."""
        idx = RouteIndex()
        mixin = MagicMock()
        h = _MinimalHandler()
        mixin._health_handler = h
        # _health_handler matches PREFIX_PATTERNS in core.py
        idx.build(mixin, [("_health_handler", _MinimalHandler)])
        prefixes = [p for p, _, _ in idx._prefix_routes]
        assert "/healthz" in prefixes or "/readyz" in prefixes or "/api/health" in prefixes


class TestGetRouteIndex:
    """Tests for global route index singleton."""

    def test_returns_route_index(self) -> None:
        idx = get_route_index()
        assert isinstance(idx, RouteIndex)

    def test_returns_same_instance(self) -> None:
        idx1 = get_route_index()
        idx2 = get_route_index()
        assert idx1 is idx2


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidateHandlerClass:
    """Tests for handler class validation."""

    def test_valid_handler(self) -> None:
        errors = validate_handler_class(_GoodHandler, "GoodHandler")
        assert errors == []

    def test_minimal_handler(self) -> None:
        errors = validate_handler_class(_MinimalHandler, "MinimalHandler")
        assert errors == []

    def test_none_handler(self) -> None:
        errors = validate_handler_class(None, "NoneHandler")
        assert len(errors) == 1
        assert "None" in errors[0]

    def test_missing_can_handle(self) -> None:
        errors = validate_handler_class(_NoCanHandle, "NoCanHandle")
        assert any("can_handle" in e for e in errors)

    def test_missing_handle(self) -> None:
        errors = validate_handler_class(_NoHandle, "NoHandle")
        assert any("handle" in e for e in errors)

    def test_non_callable_methods(self) -> None:
        errors = validate_handler_class(_NonCallableMethods, "NonCallable")
        assert any("not callable" in e for e in errors)

    def test_handler_without_routes_is_valid(self) -> None:
        """Missing ROUTES is a debug-level note, not an error."""
        errors = validate_handler_class(_MinimalHandler, "Minimal")
        assert errors == []


class TestValidateHandlerInstance:
    """Tests for handler instance validation."""

    def test_valid_instance(self) -> None:
        handler = _GoodHandler()
        errors = validate_handler_instance(handler, "Good")
        assert errors == []

    def test_none_instance(self) -> None:
        errors = validate_handler_instance(None, "None")
        assert len(errors) == 1
        assert "None" in errors[0]

    def test_broken_can_handle(self) -> None:
        handler = _BrokenCanHandle()
        errors = validate_handler_instance(handler, "Broken")
        assert len(errors) == 1
        assert "exception" in errors[0].lower() or "raised" in errors[0].lower()

    def test_non_bool_can_handle(self) -> None:
        handler = _WrongReturnType()
        errors = validate_handler_instance(handler, "WrongReturn")
        assert len(errors) == 1
        assert "non-bool" in errors[0]


class TestValidateAllHandlers:
    """Tests for full handler registry validation."""

    def test_all_valid(self) -> None:
        registry = [
            ("_good_handler", _GoodHandler),
            ("_minimal_handler", _MinimalHandler),
        ]
        results = validate_all_handlers(registry, handlers_available=True)
        assert results["status"] == "ok"
        assert len(results["valid"]) == 2
        assert len(results["invalid"]) == 0
        assert len(results["missing"]) == 0

    def test_with_none_handler(self) -> None:
        registry = [
            ("_good_handler", _GoodHandler),
            ("_missing_handler", None),
        ]
        results = validate_all_handlers(registry, handlers_available=True)
        assert len(results["valid"]) == 1
        assert len(results["missing"]) == 1

    def test_with_invalid_handler(self) -> None:
        registry = [
            ("_no_can_handle_handler", _NoCanHandle),
        ]
        results = validate_all_handlers(registry, handlers_available=True)
        assert len(results["invalid"]) == 1
        assert results["status"] == "validation_errors"

    def test_handlers_not_available(self) -> None:
        registry = [("_good_handler", _GoodHandler)]
        results = validate_all_handlers(registry, handlers_available=False)
        assert results["status"] == "imports_failed"
        assert len(results["missing"]) == 1

    def test_raise_on_error(self) -> None:
        registry = [("_no_can_handle_handler", _NoCanHandle)]
        with pytest.raises(HandlerValidationError):
            validate_all_handlers(registry, handlers_available=True, raise_on_error=True)

    def test_no_raise_when_valid(self) -> None:
        registry = [("_good_handler", _GoodHandler)]
        results = validate_all_handlers(registry, handlers_available=True, raise_on_error=True)
        assert results["status"] == "ok"

    def test_empty_registry(self) -> None:
        results = validate_all_handlers([], handlers_available=True)
        assert results["status"] == "ok"
        assert len(results["valid"]) == 0


class TestValidateHandlersOnInit:
    """Tests for post-initialization handler validation."""

    def test_all_valid(self) -> None:
        mixin = MagicMock()
        mixin._good_handler = _GoodHandler()
        registry = [("_good_handler", _GoodHandler)]
        results = validate_handlers_on_init(mixin, registry)
        assert len(results["valid"]) == 1
        assert len(results["invalid"]) == 0

    def test_not_initialized(self) -> None:
        mixin = MagicMock()
        mixin._missing_handler = None
        registry = [("_missing_handler", _MinimalHandler)]
        results = validate_handlers_on_init(mixin, registry)
        assert len(results["not_initialized"]) == 1

    def test_broken_handler(self) -> None:
        mixin = MagicMock()
        mixin._broken_handler = _BrokenCanHandle()
        registry = [("_broken_handler", _BrokenCanHandle)]
        results = validate_handlers_on_init(mixin, registry)
        assert len(results["invalid"]) == 1

    def test_mixed_results(self) -> None:
        mixin = MagicMock()
        mixin._good_handler = _GoodHandler()
        mixin._broken_handler = _BrokenCanHandle()
        mixin._missing_handler = None
        registry = [
            ("_good_handler", _GoodHandler),
            ("_broken_handler", _BrokenCanHandle),
            ("_missing_handler", _MinimalHandler),
        ]
        results = validate_handlers_on_init(mixin, registry)
        assert len(results["valid"]) == 1
        assert len(results["invalid"]) == 1
        assert len(results["not_initialized"]) == 1


class TestHandlerValidationError:
    """Tests for HandlerValidationError exception."""

    def test_is_exception(self) -> None:
        assert issubclass(HandlerValidationError, Exception)

    def test_message(self) -> None:
        err = HandlerValidationError("test error")
        assert str(err) == "test error"
