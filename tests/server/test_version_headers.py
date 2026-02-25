"""
Tests for X-API-Version header injection in ResponseHelpersMixin.

Tests cover:
- Version header injection for /api/v1/* paths
- Version header injection for /api/v2/* paths
- Default version header for /api/* paths without explicit version
- No version header for non-API paths
- Graceful exception handling when versioning module fails
- OpenAPI decorator coverage audit for handler files
"""

from __future__ import annotations

import importlib
import inspect
import logging
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.middleware.versioning import (
    APIVersion,
    CURRENT_VERSION,
    get_api_version,
)
from aragora.server.response_utils import ResponseHelpersMixin


class FakeHandler(ResponseHelpersMixin):
    """Minimal fake HTTP handler to test ResponseHelpersMixin._add_version_headers.

    Simulates the interface of BaseHTTPRequestHandler that the mixin relies on:
    - self.path: the request path
    - self.send_header(name, value): records headers sent
    """

    def __init__(self, path: str = "") -> None:
        self.path = path
        self.sent_headers: list[tuple[str, str]] = []

    def send_header(self, keyword: str, value: str) -> None:
        self.sent_headers.append((keyword, value))

    def get_header(self, name: str) -> str | None:
        """Helper: return the value of the first header matching *name*, or None."""
        for k, v in self.sent_headers:
            if k == name:
                return v
        return None

    def header_names(self) -> list[str]:
        """Helper: return all header names that were sent."""
        return [k for k, _ in self.sent_headers]


# ---------------------------------------------------------------------------
# Test 1: X-API-Version is "v1" for /api/v1/* paths
# ---------------------------------------------------------------------------
class TestVersionHeaderV1:
    """X-API-Version header should be 'v1' for /api/v1/* paths."""

    def test_v1_debates(self) -> None:
        handler = FakeHandler("/api/v1/debates")
        handler._add_version_headers()
        assert handler.get_header("X-API-Version") == "v1"

    def test_v1_nested_path(self) -> None:
        handler = FakeHandler("/api/v1/agents/123/config")
        handler._add_version_headers()
        assert handler.get_header("X-API-Version") == "v1"

    def test_v1_root(self) -> None:
        handler = FakeHandler("/api/v1/")
        handler._add_version_headers()
        assert handler.get_header("X-API-Version") == "v1"


# ---------------------------------------------------------------------------
# Test 2: X-API-Version is "v2" for /api/v2/* paths
# ---------------------------------------------------------------------------
class TestVersionHeaderV2:
    """X-API-Version header should be 'v2' for /api/v2/* paths."""

    def test_v2_debates(self) -> None:
        handler = FakeHandler("/api/v2/debates")
        handler._add_version_headers()
        assert handler.get_header("X-API-Version") == "v2"

    def test_v2_nested_path(self) -> None:
        handler = FakeHandler("/api/v2/knowledge/search?q=test")
        handler._add_version_headers()
        assert handler.get_header("X-API-Version") == "v2"

    def test_v2_root(self) -> None:
        handler = FakeHandler("/api/v2/")
        handler._add_version_headers()
        assert handler.get_header("X-API-Version") == "v2"


# ---------------------------------------------------------------------------
# Test 3: X-API-Version defaults to current version for /api/* without version
# ---------------------------------------------------------------------------
class TestVersionHeaderDefault:
    """X-API-Version should default to CURRENT_VERSION for /api/* without explicit version."""

    def test_api_without_version_defaults_to_current(self) -> None:
        handler = FakeHandler("/api/debates")
        handler._add_version_headers()
        assert handler.get_header("X-API-Version") == CURRENT_VERSION.value

    def test_api_without_version_is_v2(self) -> None:
        """Current version is V2 at time of writing."""
        handler = FakeHandler("/api/status")
        handler._add_version_headers()
        assert handler.get_header("X-API-Version") == "v2"

    def test_api_with_invalid_version_defaults(self) -> None:
        """An invalid version segment like /api/v99/ should fall back to CURRENT_VERSION."""
        handler = FakeHandler("/api/v99/debates")
        handler._add_version_headers()
        assert handler.get_header("X-API-Version") == CURRENT_VERSION.value


# ---------------------------------------------------------------------------
# Test 4: No X-API-Version header for non-API paths
# ---------------------------------------------------------------------------
class TestVersionHeaderNonApiPaths:
    """Non-API paths should NOT receive an X-API-Version header."""

    def test_healthz(self) -> None:
        handler = FakeHandler("/healthz")
        handler._add_version_headers()
        assert handler.get_header("X-API-Version") is None
        assert len(handler.sent_headers) == 0

    def test_static_file(self) -> None:
        handler = FakeHandler("/static/app.js")
        handler._add_version_headers()
        assert handler.get_header("X-API-Version") is None

    def test_root_path(self) -> None:
        handler = FakeHandler("/")
        handler._add_version_headers()
        assert handler.get_header("X-API-Version") is None

    def test_empty_path(self) -> None:
        handler = FakeHandler("")
        handler._add_version_headers()
        assert handler.get_header("X-API-Version") is None

    def test_no_path_attribute(self) -> None:
        """If 'path' attribute is missing entirely, no header should be set."""
        handler = FakeHandler()
        del handler.path
        handler._add_version_headers()
        assert len(handler.sent_headers) == 0

    def test_metrics_path(self) -> None:
        handler = FakeHandler("/metrics")
        handler._add_version_headers()
        assert handler.get_header("X-API-Version") is None

    def test_ws_path(self) -> None:
        handler = FakeHandler("/ws/debate/123")
        handler._add_version_headers()
        assert handler.get_header("X-API-Version") is None


# ---------------------------------------------------------------------------
# Test 5: Graceful exception handling
# ---------------------------------------------------------------------------
class TestVersionHeaderExceptionHandling:
    """When the versioning module raises, _add_version_headers should log and continue."""

    def test_exception_in_get_api_version_is_swallowed(self) -> None:
        """If get_api_version raises, no header is set and no exception propagates."""
        handler = FakeHandler("/api/v1/debates")
        # The lazy import inside _add_version_headers pulls from the source module,
        # so we patch at the source: aragora.server.middleware.versioning.get_api_version
        with patch(
            "aragora.server.middleware.versioning.get_api_version",
            side_effect=ValueError("versioning module broken"),
        ):
            # Should NOT raise
            handler._add_version_headers()
        assert handler.get_header("X-API-Version") is None

    def test_exception_is_logged_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """The swallowed exception should be logged at DEBUG level."""
        handler = FakeHandler("/api/v2/debates")
        with patch(
            "aragora.server.middleware.versioning.get_api_version",
            side_effect=ValueError("bad version"),
        ):
            with caplog.at_level(logging.DEBUG, logger="aragora.server.response_utils"):
                handler._add_version_headers()
        assert any("Version header injection failed" in msg for msg in caplog.messages)

    def test_import_error_is_handled(self) -> None:
        """Even if the import inside the method fails, it should be caught."""
        handler = FakeHandler("/api/v1/test")
        # Simulate an ImportError by temporarily making the import fail.
        # We patch the source module function to raise ImportError.
        with patch(
            "aragora.server.middleware.versioning.get_api_version",
            side_effect=ImportError("no module named versioning"),
        ):
            handler._add_version_headers()
        assert handler.get_header("X-API-Version") is None

    def test_send_header_exception_is_handled(self) -> None:
        """If send_header itself raises, it should be caught by the outer try/except."""
        handler = FakeHandler("/api/v1/debates")
        handler.send_header = MagicMock(side_effect=OSError("broken pipe"))
        handler._add_version_headers()
        # No exception propagated -- test passes if we get here

    def test_only_one_header_per_call(self) -> None:
        """_add_version_headers should send exactly one header for API paths."""
        handler = FakeHandler("/api/v2/debates")
        handler._add_version_headers()
        version_headers = [k for k, _ in handler.sent_headers if k == "X-API-Version"]
        assert len(version_headers) == 1


# ---------------------------------------------------------------------------
# Test 6: OpenAPI decorator coverage audit
# ---------------------------------------------------------------------------

# A selection of handler files known to have classes with can_handle/handle
# that SHOULD also use @api_endpoint decorators.
_HANDLERS_EXPECTED_TO_HAVE_API_ENDPOINT = [
    "aragora.server.handlers.receipts",
    "aragora.server.handlers.gauntlet.handler",
    "aragora.server.handlers.debates.handler",
    "aragora.server.handlers.debates.graph_debates",
    "aragora.server.handlers.admin.handler",
]

# Handler files known to NOT require @api_endpoint (e.g., base classes, mixins,
# internal utilities, type definitions).
_HANDLER_EXEMPTIONS = {
    "aragora.server.handlers.base",
    "aragora.server.handlers.interface",
    "aragora.server.handlers.types",
    "aragora.server.handlers.mixins",
    "aragora.server.handlers.__init__",
    "aragora.server.handlers._registry",
    "aragora.server.handlers.composite",
    "aragora.server.handlers.exceptions",
    "aragora.server.handlers.utils.routing",
    "aragora.server.handlers.utils.responses",
}


def _module_has_handler_class(module: types.ModuleType) -> bool:
    """Return True if the module defines a class with `can_handle` or `handle` methods."""
    for _name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue  # skip imported classes
        if hasattr(obj, "can_handle") or hasattr(obj, "handle"):
            return True
    return False


def _module_has_api_endpoint_decorator(module: types.ModuleType) -> bool:
    """Return True if any function/method in the module has an @api_endpoint marker.

    The openapi_decorator.api_endpoint sets ``_openapi`` on the wrapper.
    The api_decorators.api_endpoint sets ``_api_metadata`` on the function.
    We check for both to cover either decorator being used.
    """
    marker_attrs = ("_openapi", "_api_metadata")
    for _name, obj in inspect.getmembers(module):
        if callable(obj) and any(hasattr(obj, attr) for attr in marker_attrs):
            return True
        # Also check methods inside classes defined in this module
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            for _method_name, method in inspect.getmembers(obj, predicate=inspect.isfunction):
                if any(hasattr(method, attr) for attr in marker_attrs):
                    return True
    return False


class TestOpenAPIDecoratorCoverageAudit:
    """Spot-check that handler files with can_handle/handle also use @api_endpoint.

    This is not an exhaustive audit -- it checks a known set of handler modules
    that are expected to have @api_endpoint decorators applied.
    """

    @pytest.mark.parametrize("module_path", _HANDLERS_EXPECTED_TO_HAVE_API_ENDPOINT)
    def test_known_handlers_have_api_endpoint(self, module_path: str) -> None:
        """Known handler modules with can_handle/handle should use @api_endpoint."""
        module = importlib.import_module(module_path)

        assert _module_has_handler_class(module), (
            f"{module_path} should define a handler class with can_handle or handle"
        )
        assert _module_has_api_endpoint_decorator(module), (
            f"{module_path} has handler class(es) but no @api_endpoint decorator usage. "
            "All handler methods that serve API endpoints should be decorated with "
            "@api_endpoint for OpenAPI schema generation."
        )

    def test_receipts_handler_has_api_endpoint(self) -> None:
        """ReceiptsHandler specifically should have @api_endpoint decorators."""
        from aragora.server.handlers import receipts

        assert _module_has_api_endpoint_decorator(receipts), (
            "receipts handler should use @api_endpoint"
        )

    def test_api_endpoint_decorator_exists_and_importable(self) -> None:
        """The @api_endpoint decorator itself should be importable."""
        from aragora.server.handlers.openapi_decorator import api_endpoint

        assert callable(api_endpoint)

    def test_openapi_api_endpoint_decorator_sets_openapi_attr(self) -> None:
        """@api_endpoint from openapi_decorator should attach _openapi to decorated functions."""
        from aragora.server.handlers.openapi_decorator import api_endpoint

        @api_endpoint(
            path="/api/v1/test",
            method="GET",
            summary="Test endpoint",
            tags=["Test"],
        )
        def dummy_handler() -> None:
            pass

        assert hasattr(dummy_handler, "_openapi"), (
            "openapi_decorator.api_endpoint should set _openapi attribute"
        )
        endpoint = dummy_handler._openapi  # type: ignore[attr-defined]
        assert endpoint.path == "/api/v1/test"
        assert endpoint.method == "GET"
        assert endpoint.summary == "Test endpoint"

    def test_api_decorators_api_endpoint_sets_metadata(self) -> None:
        """@api_endpoint from api_decorators should attach _api_metadata."""
        from aragora.server.handlers.api_decorators import api_endpoint

        @api_endpoint(
            method="POST",
            path="/api/v2/items",
            summary="Create item",
        )
        def dummy_handler() -> None:
            pass

        assert hasattr(dummy_handler, "_api_metadata"), (
            "api_decorators.api_endpoint should set _api_metadata attribute"
        )
        meta = dummy_handler._api_metadata  # type: ignore[attr-defined]
        assert meta["path"] == "/api/v2/items"
        assert meta["method"] == "POST"
        assert meta["summary"] == "Create item"


# ---------------------------------------------------------------------------
# Additional integration-style tests
# ---------------------------------------------------------------------------
class TestGetApiVersionConsistency:
    """Verify that get_api_version (used by _add_version_headers) behaves as expected."""

    def test_v1_extraction(self) -> None:
        assert get_api_version("/api/v1/debates") == APIVersion.V1

    def test_v2_extraction(self) -> None:
        assert get_api_version("/api/v2/debates") == APIVersion.V2

    def test_unversioned_api_path_returns_current(self) -> None:
        assert get_api_version("/api/debates") == CURRENT_VERSION

    def test_non_api_path_returns_current(self) -> None:
        assert get_api_version("/healthz") == CURRENT_VERSION

    def test_case_insensitive_version(self) -> None:
        """Version segment matching should be case-insensitive (lowercase internally)."""
        assert get_api_version("/api/V1/debates") == APIVersion.V1

    def test_trailing_slash(self) -> None:
        assert get_api_version("/api/v2/") == APIVersion.V2
