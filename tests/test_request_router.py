"""Tests for request router."""

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass

from aragora.server.router import RequestRouter, Route


@dataclass
class MockHandlerResult:
    """Mock handler result."""

    status_code: int = 200
    body: bytes = b"{}"
    content_type: str = "application/json"


class MockHandler:
    """Mock handler for testing."""

    ROUTES = ["/api/test", "/api/test/{id}"]

    def __init__(self, server_context: dict):
        self.ctx = server_context
        self.handled_requests = []

    def can_handle(self, path: str) -> bool:
        return path.startswith("/api/test")

    def handle(self, path: str, query_params: dict, handler=None):
        self.handled_requests.append(("GET", path, query_params))
        return MockHandlerResult(status_code=200)

    def handle_post(self, path: str, query_params: dict, handler=None):
        self.handled_requests.append(("POST", path, query_params))
        return MockHandlerResult(status_code=201)


class MockHandlerNoRoutes:
    """Handler that uses can_handle instead of ROUTES."""

    def __init__(self, server_context: dict):
        self.ctx = server_context

    def can_handle(self, path: str) -> bool:
        return path == "/api/fallback"

    def handle(self, path: str, query_params: dict, handler=None):
        return MockHandlerResult(status_code=200)


class TestRoute:
    """Tests for Route class."""

    def test_matches_exact_path(self):
        """Route matches exact path."""
        import re

        route = Route(
            pattern=re.compile(r"^/api/test$"),
            handler=MagicMock(),
            methods={"GET"},
        )
        matches, params = route.matches("/api/test", "GET")
        assert matches is True
        assert params == {}

    def test_no_match_wrong_method(self):
        """Route doesn't match wrong method."""
        import re

        route = Route(
            pattern=re.compile(r"^/api/test$"),
            handler=MagicMock(),
            methods={"GET"},
        )
        matches, _ = route.matches("/api/test", "POST")
        assert matches is False

    def test_no_match_wrong_path(self):
        """Route doesn't match wrong path."""
        import re

        route = Route(
            pattern=re.compile(r"^/api/test$"),
            handler=MagicMock(),
            methods={"GET"},
        )
        matches, _ = route.matches("/api/other", "GET")
        assert matches is False

    def test_extracts_path_params(self):
        """Route extracts path parameters."""
        import re

        route = Route(
            pattern=re.compile(r"^/api/items/(?P<id>[^/]+)$"),
            handler=MagicMock(),
            methods={"GET"},
        )
        matches, params = route.matches("/api/items/123", "GET")
        assert matches is True
        assert params == {"id": "123"}


class TestRequestRouter:
    """Tests for RequestRouter."""

    @pytest.fixture
    def router(self):
        """Create router instance."""
        return RequestRouter()

    @pytest.fixture
    def mock_handler(self):
        """Create mock handler."""
        return MockHandler({})

    def test_register_handler(self, router, mock_handler):
        """Can register a handler."""
        router.register(mock_handler)
        assert mock_handler in router._handlers
        assert len(router._routes) == 2  # Two routes from ROUTES

    def test_dispatch_get(self, router, mock_handler):
        """Dispatches GET requests."""
        router.register(mock_handler)
        result = router.dispatch("GET", "/api/test", {"limit": ["10"]})
        assert result is not None
        assert result.status_code == 200
        assert ("GET", "/api/test", {"limit": ["10"]}) in mock_handler.handled_requests

    def test_dispatch_post(self, router, mock_handler):
        """Dispatches POST requests."""
        router.register(mock_handler)
        result = router.dispatch("POST", "/api/test", {})
        assert result is not None
        assert result.status_code == 201
        assert ("POST", "/api/test", {}) in mock_handler.handled_requests

    def test_dispatch_with_path_param(self, router, mock_handler):
        """Dispatches requests with path parameters."""
        router.register(mock_handler)
        result = router.dispatch("GET", "/api/test/123", {})
        assert result is not None

    def test_dispatch_unmatched_path(self, router, mock_handler):
        """Returns None for unmatched paths."""
        router.register(mock_handler)
        result = router.dispatch("GET", "/api/other", {})
        assert result is None

    def test_dispatch_fallback_to_can_handle(self, router):
        """Falls back to can_handle for handlers without ROUTES."""
        handler = MockHandlerNoRoutes({})
        router.register(handler)
        result = router.dispatch("GET", "/api/fallback", {})
        assert result is not None

    def test_get_all_routes(self, router, mock_handler):
        """Returns list of all registered routes."""
        router.register(mock_handler)
        routes = router.get_all_routes()
        assert len(routes) == 2
        assert any("/api/test" in r["pattern"] for r in routes)

    def test_get_handler_for_path(self, router, mock_handler):
        """Returns correct handler for path."""
        router.register(mock_handler)
        handler = router.get_handler_for_path("/api/test")
        assert handler is mock_handler

    def test_get_handler_for_unknown_path(self, router, mock_handler):
        """Returns None for unknown path."""
        router.register(mock_handler)
        handler = router.get_handler_for_path("/api/unknown")
        assert handler is None

    def test_multiple_handlers(self, router):
        """Routes to correct handler when multiple registered."""
        handler1 = MockHandler({})
        handler2 = MockHandlerNoRoutes({})

        router.register(handler1)
        router.register(handler2)

        # Should route to handler1
        result1 = router.dispatch("GET", "/api/test", {})
        assert result1 is not None
        assert ("GET", "/api/test", {}) in handler1.handled_requests

        # Should route to handler2 via fallback
        result2 = router.dispatch("GET", "/api/fallback", {})
        assert result2 is not None

    def test_handler_error_returns_none(self, router):
        """Returns None when handler raises error."""
        handler = MagicMock()
        handler.ROUTES = ["/api/error"]
        handler.handle.side_effect = ValueError("Test error")
        handler.can_handle.return_value = True

        router.register(handler)
        result = router.dispatch("GET", "/api/error", {})
        assert result is None  # Error should return None

    def test_method_detection(self, router):
        """Detects available methods from handler."""
        handler = MockHandler({})
        router.register(handler)

        # GET should work
        result_get = router.dispatch("GET", "/api/test", {})
        assert result_get is not None

        # POST should work (handle_post exists)
        result_post = router.dispatch("POST", "/api/test", {})
        assert result_post is not None

        # DELETE should not work (no handle_delete)
        result_delete = router.dispatch("DELETE", "/api/test", {})
        assert result_delete is None
