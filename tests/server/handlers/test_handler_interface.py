"""
Tests for aragora.server.handlers.interface.

Tests cover:
1. HandlerInterface protocol - runtime checkable behavior
2. AuthenticatedHandlerInterface protocol - runtime checkable behavior
3. PaginatedHandlerInterface protocol - runtime checkable behavior
4. CachedHandlerInterface protocol - runtime checkable behavior
5. StorageAccessInterface protocol - runtime checkable behavior
6. is_handler() factory function
7. is_authenticated_handler() factory function
8. TypedDict structures (HandlerResult, RouteConfig, etc.)
"""

from __future__ import annotations

from typing import Any
from collections.abc import Awaitable

import pytest

from aragora.server.handlers.interface import (
    AuthenticatedHandlerInterface,
    CachedHandlerInterface,
    HandlerInterface,
    HandlerRegistration,
    HandlerResult,
    MaybeAsyncHandlerResult,
    MinimalServerContext,
    PaginatedHandlerInterface,
    RouteConfig,
    StorageAccessInterface,
    is_authenticated_handler,
    is_handler,
)


# =============================================================================
# Test Implementations
# =============================================================================


class _GoodHandler:
    """A class that implements HandlerInterface properly."""

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


class _PartialHandler:
    """A class missing some handler methods."""

    def handle(self, path, query_params, handler):
        return None


class _NotAHandler:
    """A class that does not implement HandlerInterface."""

    def do_something(self):
        pass


class _AuthHandler:
    """A class implementing AuthenticatedHandlerInterface."""

    def get_current_user(self, handler):
        return None

    def require_auth_or_error(self, handler):
        return None, None


class _PaginatedHandler:
    """A class implementing PaginatedHandlerInterface."""

    def get_pagination(self, query_params, default_limit=None, max_limit=None):
        return (20, 0)

    def paginated_response(self, items, total, limit, offset, items_key="items"):
        return HandlerResult(body=b"{}", content_type="application/json", status=200)


class _CachedHandler:
    """A class implementing CachedHandlerInterface."""

    def cached_response(self, cache_key, ttl_seconds, generator):
        return generator()


class _StorageAccess:
    """A class implementing StorageAccessInterface."""

    def get_storage(self):
        return None

    def get_elo_system(self):
        return None


# =============================================================================
# is_handler Tests
# =============================================================================


class TestIsHandler:
    """Test is_handler() factory function."""

    def test_good_handler_detected(self):
        handler = _GoodHandler()
        assert is_handler(handler) is True

    def test_not_a_handler_rejected(self):
        obj = _NotAHandler()
        assert is_handler(obj) is False

    def test_partial_handler_rejected(self):
        """A handler missing some methods should not satisfy the protocol."""
        obj = _PartialHandler()
        assert is_handler(obj) is False

    def test_none_rejected(self):
        assert is_handler(None) is False

    def test_string_rejected(self):
        assert is_handler("not a handler") is False

    def test_dict_rejected(self):
        assert is_handler({}) is False

    def test_integer_rejected(self):
        assert is_handler(42) is False


# =============================================================================
# is_authenticated_handler Tests
# =============================================================================


class TestIsAuthenticatedHandler:
    """Test is_authenticated_handler() factory function."""

    def test_auth_handler_detected(self):
        handler = _AuthHandler()
        assert is_authenticated_handler(handler) is True

    def test_plain_handler_rejected(self):
        handler = _GoodHandler()
        assert is_authenticated_handler(handler) is False

    def test_not_a_handler_rejected(self):
        obj = _NotAHandler()
        assert is_authenticated_handler(obj) is False

    def test_none_rejected(self):
        assert is_authenticated_handler(None) is False


# =============================================================================
# Protocol Runtime Checks
# =============================================================================


class TestHandlerInterfaceProtocol:
    """Test HandlerInterface as a runtime_checkable Protocol."""

    def test_isinstance_check_works(self):
        handler = _GoodHandler()
        assert isinstance(handler, HandlerInterface)

    def test_isinstance_check_fails_for_non_handler(self):
        obj = _NotAHandler()
        assert not isinstance(obj, HandlerInterface)


class TestAuthenticatedHandlerInterfaceProtocol:
    """Test AuthenticatedHandlerInterface as a runtime_checkable Protocol."""

    def test_isinstance_check_works(self):
        handler = _AuthHandler()
        assert isinstance(handler, AuthenticatedHandlerInterface)

    def test_isinstance_check_fails_for_non_handler(self):
        assert not isinstance(_NotAHandler(), AuthenticatedHandlerInterface)


class TestPaginatedHandlerInterfaceProtocol:
    """Test PaginatedHandlerInterface as a runtime_checkable Protocol."""

    def test_isinstance_check_works(self):
        handler = _PaginatedHandler()
        assert isinstance(handler, PaginatedHandlerInterface)

    def test_isinstance_check_fails_for_non_handler(self):
        assert not isinstance(_NotAHandler(), PaginatedHandlerInterface)


class TestCachedHandlerInterfaceProtocol:
    """Test CachedHandlerInterface as a runtime_checkable Protocol."""

    def test_isinstance_check_works(self):
        handler = _CachedHandler()
        assert isinstance(handler, CachedHandlerInterface)

    def test_isinstance_check_fails_for_non_handler(self):
        assert not isinstance(_NotAHandler(), CachedHandlerInterface)


class TestStorageAccessInterfaceProtocol:
    """Test StorageAccessInterface Protocol definition."""

    def test_storage_access_has_get_storage(self):
        """StorageAccessInterface should define get_storage."""
        obj = _StorageAccess()
        assert hasattr(obj, "get_storage")
        assert callable(obj.get_storage)

    def test_storage_access_has_get_elo_system(self):
        """StorageAccessInterface should define get_elo_system."""
        obj = _StorageAccess()
        assert hasattr(obj, "get_elo_system")
        assert callable(obj.get_elo_system)

    def test_non_handler_lacks_storage_methods(self):
        obj = _NotAHandler()
        assert not hasattr(obj, "get_storage")
        assert not hasattr(obj, "get_elo_system")


# =============================================================================
# TypedDict Structure Tests
# =============================================================================


class TestTypedDicts:
    """Test that TypedDict structures can be instantiated."""

    def test_handler_result_creation(self):
        """HandlerResult TypedDict should accept expected keys."""
        result: HandlerResult = {"body": b"data", "content_type": "application/json"}
        assert result["body"] == b"data"
        assert result["content_type"] == "application/json"

    def test_route_config_creation(self):
        """RouteConfig TypedDict should accept expected keys."""
        config: RouteConfig = {
            "path_pattern": "/api/test",
            "methods": ["GET", "POST"],
            "requires_auth": True,
        }
        assert config["path_pattern"] == "/api/test"
        assert config["methods"] == ["GET", "POST"]

    def test_handler_registration_creation(self):
        """HandlerRegistration TypedDict should accept expected keys."""
        reg: HandlerRegistration = {
            "handler_class": _GoodHandler,
            "routes": [],
            "lazy": True,
        }
        assert reg["handler_class"] is _GoodHandler
        assert reg["lazy"] is True

    def test_minimal_server_context_creation(self):
        """MinimalServerContext TypedDict should accept expected keys."""
        ctx: MinimalServerContext = {"storage": None, "user_store": None}
        assert ctx["storage"] is None
