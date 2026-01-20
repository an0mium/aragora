"""
Tests for handler interface module.

Tests cover:
- HandlerResult TypedDict
- HandlerInterface protocol
- AuthenticatedHandlerInterface protocol
- PaginatedHandlerInterface protocol
- CachedHandlerInterface protocol
- is_handler and is_authenticated_handler functions
"""

from typing import Any, Dict, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from aragora.server.handlers.interface import (
    HandlerResult,
    HandlerInterface,
    AuthenticatedHandlerInterface,
    PaginatedHandlerInterface,
    CachedHandlerInterface,
    MinimalServerContext,
    RouteConfig,
    HandlerRegistration,
    is_handler,
    is_authenticated_handler,
)


class TestHandlerResultType:
    """Tests for HandlerResult TypedDict."""

    def test_can_create_full_result(self):
        """Test creating a full result with all fields."""
        result: HandlerResult = {
            "body": b"test body",
            "content_type": "application/json",
            "status": 200,
            "headers": {"X-Custom": "value"},
        }
        assert result["body"] == b"test body"
        assert result["content_type"] == "application/json"
        assert result["status"] == 200
        assert result["headers"]["X-Custom"] == "value"

    def test_can_create_partial_result(self):
        """Test creating result with partial fields (total=False)."""
        result: HandlerResult = {
            "body": b"test",
            "status": 200,
        }
        assert result["body"] == b"test"
        assert result["status"] == 200


class TestHandlerInterfaceProtocol:
    """Tests for HandlerInterface protocol checking."""

    def test_object_with_all_methods_is_handler(self):
        """Test that object implementing all methods is recognized."""

        class FullHandler:
            def handle(
                self, path: str, query_params: Dict[str, Any], handler: Any
            ) -> Optional[HandlerResult]:
                return None

            def handle_post(
                self, path: str, query_params: Dict[str, Any], handler: Any
            ) -> Optional[HandlerResult]:
                return None

            def handle_delete(
                self, path: str, query_params: Dict[str, Any], handler: Any
            ) -> Optional[HandlerResult]:
                return None

            def handle_patch(
                self, path: str, query_params: Dict[str, Any], handler: Any
            ) -> Optional[HandlerResult]:
                return None

            def handle_put(
                self, path: str, query_params: Dict[str, Any], handler: Any
            ) -> Optional[HandlerResult]:
                return None

        handler = FullHandler()
        assert is_handler(handler)
        assert isinstance(handler, HandlerInterface)

    def test_object_with_partial_methods_not_handler(self):
        """Test that object missing methods is not recognized."""

        class PartialHandler:
            def handle(
                self, path: str, query_params: Dict[str, Any], handler: Any
            ) -> Optional[HandlerResult]:
                return None

            # Missing other handle_* methods

        handler = PartialHandler()
        # Protocol check depends on runtime_checkable behavior
        # Objects with only some methods should not pass isinstance
        assert not isinstance(handler, HandlerInterface)

    def test_mock_handler_can_be_made_compatible(self):
        """Test creating a mock that matches interface."""
        mock_handler = MagicMock(
            spec=["handle", "handle_post", "handle_delete", "handle_patch", "handle_put"]
        )
        mock_handler.handle.return_value = None
        mock_handler.handle_post.return_value = None
        mock_handler.handle_delete.return_value = None
        mock_handler.handle_patch.return_value = None
        mock_handler.handle_put.return_value = None

        assert isinstance(mock_handler, HandlerInterface)


class TestAuthenticatedHandlerInterfaceProtocol:
    """Tests for AuthenticatedHandlerInterface protocol."""

    def test_object_with_auth_methods_is_authenticated_handler(self):
        """Test object with auth methods is recognized."""

        class AuthHandler:
            def get_current_user(self, handler: Any) -> Optional[Any]:
                return None

            def require_auth_or_error(
                self, handler: Any
            ) -> Tuple[Optional[Any], Optional[HandlerResult]]:
                return (None, None)

        handler = AuthHandler()
        assert is_authenticated_handler(handler)
        assert isinstance(handler, AuthenticatedHandlerInterface)

    def test_object_without_auth_methods_not_authenticated(self):
        """Test object without auth methods is not recognized."""

        class NonAuthHandler:
            def handle(
                self, path: str, query_params: Dict[str, Any], handler: Any
            ) -> Optional[HandlerResult]:
                return None

        handler = NonAuthHandler()
        assert not is_authenticated_handler(handler)
        assert not isinstance(handler, AuthenticatedHandlerInterface)


class TestPaginatedHandlerInterfaceProtocol:
    """Tests for PaginatedHandlerInterface protocol."""

    def test_object_with_pagination_methods_matches(self):
        """Test object with pagination methods matches protocol."""

        class PaginatedHandler:
            def get_pagination(
                self,
                query_params: Dict[str, Any],
                default_limit: Optional[int] = None,
                max_limit: Optional[int] = None,
            ) -> Tuple[int, int]:
                return (10, 0)

            def paginated_response(
                self,
                items: list,
                total: int,
                limit: int,
                offset: int,
                items_key: str = "items",
            ) -> HandlerResult:
                return {"body": b"{}", "status": 200}

        handler = PaginatedHandler()
        assert isinstance(handler, PaginatedHandlerInterface)


class TestCachedHandlerInterfaceProtocol:
    """Tests for CachedHandlerInterface protocol."""

    def test_object_with_cache_method_matches(self):
        """Test object with cached_response method matches protocol."""

        class CachedHandler:
            def cached_response(
                self,
                cache_key: str,
                ttl_seconds: float,
                generator: Any,
            ) -> Any:
                return generator()

        handler = CachedHandler()
        assert isinstance(handler, CachedHandlerInterface)


class TestIsHandlerFunction:
    """Tests for is_handler utility function."""

    def test_returns_true_for_handler(self):
        """Test returns True for valid handler."""

        class ValidHandler:
            def handle(self, path, query_params, handler):
                return None

            def handle_post(self, path, query_params, handler):
                return None

            def handle_delete(self, path, query_params, handler):
                return None

            def handle_patch(self, path, query_params, handler):
                return None

            def handle_put(self, path, query_params, handler):
                return None

        assert is_handler(ValidHandler())

    def test_returns_false_for_non_handler(self):
        """Test returns False for non-handler."""
        assert not is_handler("not a handler")
        assert not is_handler(123)
        assert not is_handler(None)
        assert not is_handler({})


class TestIsAuthenticatedHandlerFunction:
    """Tests for is_authenticated_handler utility function."""

    def test_returns_true_for_auth_handler(self):
        """Test returns True for authenticated handler."""

        class AuthHandler:
            def get_current_user(self, handler):
                return None

            def require_auth_or_error(self, handler):
                return (None, None)

        assert is_authenticated_handler(AuthHandler())

    def test_returns_false_for_non_auth_handler(self):
        """Test returns False for non-authenticated handler."""
        assert not is_authenticated_handler("not a handler")
        assert not is_authenticated_handler(123)
        assert not is_authenticated_handler(None)


class TestMinimalServerContext:
    """Tests for MinimalServerContext TypedDict."""

    def test_can_create_full_context(self):
        """Test creating full server context."""
        context: MinimalServerContext = {
            "storage": MagicMock(),
            "user_store": MagicMock(),
            "elo_system": MagicMock(),
        }
        assert "storage" in context
        assert "user_store" in context
        assert "elo_system" in context

    def test_can_create_partial_context(self):
        """Test creating partial context (total=False)."""
        context: MinimalServerContext = {
            "storage": MagicMock(),
        }
        assert "storage" in context


class TestRouteConfig:
    """Tests for RouteConfig TypedDict."""

    def test_can_create_route_config(self):
        """Test creating route config."""
        config: RouteConfig = {
            "path_pattern": "/api/test",
            "methods": ["GET", "POST"],
            "handler_class": object,
            "requires_auth": True,
            "rate_limit": 100,
        }
        assert config["path_pattern"] == "/api/test"
        assert config["methods"] == ["GET", "POST"]
        assert config["requires_auth"] is True


class TestHandlerRegistration:
    """Tests for HandlerRegistration TypedDict."""

    def test_can_create_registration(self):
        """Test creating handler registration."""
        registration: HandlerRegistration = {
            "handler_class": object,
            "routes": [{"path_pattern": "/test"}],
            "lazy": True,
        }
        assert registration["handler_class"] == object
        assert len(registration["routes"]) == 1
        assert registration["lazy"] is True


class TestProtocolComposition:
    """Tests for combining multiple protocols."""

    def test_handler_can_implement_multiple_protocols(self):
        """Test that a handler can implement multiple protocols."""

        class MultiProtocolHandler:
            # HandlerInterface
            def handle(self, path, query_params, handler):
                return None

            def handle_post(self, path, query_params, handler):
                return None

            def handle_delete(self, path, query_params, handler):
                return None

            def handle_patch(self, path, query_params, handler):
                return None

            def handle_put(self, path, query_params, handler):
                return None

            # AuthenticatedHandlerInterface
            def get_current_user(self, handler):
                return None

            def require_auth_or_error(self, handler):
                return (None, None)

            # PaginatedHandlerInterface
            def get_pagination(self, query_params, default_limit=None, max_limit=None):
                return (10, 0)

            def paginated_response(self, items, total, limit, offset, items_key="items"):
                return {"body": b"{}", "status": 200}

        handler = MultiProtocolHandler()

        assert is_handler(handler)
        assert is_authenticated_handler(handler)
        assert isinstance(handler, PaginatedHandlerInterface)
