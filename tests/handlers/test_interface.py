"""Tests for aragora/server/handlers/interface.py.

Comprehensive coverage of all interface definitions in the handler interface module:
1. HandlerResult TypedDict - field access, totality, instantiation
2. MaybeAsyncHandlerResult type alias - union semantics
3. HandlerInterface Protocol - runtime_checkable behavior, method signatures
4. AuthenticatedHandlerInterface Protocol - auth method contracts
5. PaginatedHandlerInterface Protocol - pagination method contracts
6. CachedHandlerInterface Protocol - caching method contracts
7. StorageAccessInterface Protocol - storage access contracts
8. MinimalServerContext TypedDict - field access, totality
9. RouteConfig TypedDict - field access, totality
10. HandlerRegistration TypedDict - field access
11. is_handler() factory function - type checking
12. is_authenticated_handler() factory function - type checking
13. __all__ exports - completeness and importability
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Protocol, get_type_hints
from collections.abc import Awaitable
from unittest.mock import MagicMock

import pytest

from aragora.server.handlers.interface import (
    # Result types
    HandlerResult,
    MaybeAsyncHandlerResult,
    # Protocols
    HandlerInterface,
    AuthenticatedHandlerInterface,
    PaginatedHandlerInterface,
    CachedHandlerInterface,
    StorageAccessInterface,
    # Context types
    MinimalServerContext,
    # Registration types
    RouteConfig,
    HandlerRegistration,
    # Factory functions
    is_handler,
    is_authenticated_handler,
)


# =============================================================================
# Helper Classes for Protocol Tests
# =============================================================================


class _CompleteHandler:
    """Implements all required HandlerInterface methods (sync)."""

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


class _AsyncCompleteHandler:
    """Implements all HandlerInterface methods as async."""

    async def handle(self, path, query_params, handler):
        return None

    async def handle_post(self, path, query_params, handler):
        return None

    async def handle_delete(self, path, query_params, handler):
        return None

    async def handle_patch(self, path, query_params, handler):
        return None

    async def handle_put(self, path, query_params, handler):
        return None


class _HandleOnlyHandler:
    """Only implements the handle method."""

    def handle(self, path, query_params, handler):
        return None


class _MissingPutHandler:
    """Missing handle_put method."""

    def handle(self, path, query_params, handler):
        return None

    def handle_post(self, path, query_params, handler):
        return None

    def handle_delete(self, path, query_params, handler):
        return None

    def handle_patch(self, path, query_params, handler):
        return None


class _MissingDeleteHandler:
    """Missing handle_delete method."""

    def handle(self, path, query_params, handler):
        return None

    def handle_post(self, path, query_params, handler):
        return None

    def handle_patch(self, path, query_params, handler):
        return None

    def handle_put(self, path, query_params, handler):
        return None


class _MissingPatchHandler:
    """Missing handle_patch method."""

    def handle(self, path, query_params, handler):
        return None

    def handle_post(self, path, query_params, handler):
        return None

    def handle_delete(self, path, query_params, handler):
        return None

    def handle_put(self, path, query_params, handler):
        return None


class _MissingPostHandler:
    """Missing handle_post method."""

    def handle(self, path, query_params, handler):
        return None

    def handle_delete(self, path, query_params, handler):
        return None

    def handle_patch(self, path, query_params, handler):
        return None

    def handle_put(self, path, query_params, handler):
        return None


class _MissingHandleMethod:
    """Missing the primary handle method."""

    def handle_post(self, path, query_params, handler):
        return None

    def handle_delete(self, path, query_params, handler):
        return None

    def handle_patch(self, path, query_params, handler):
        return None

    def handle_put(self, path, query_params, handler):
        return None


class _EmptyClass:
    """Empty class with no methods."""

    pass


class _WrongSignatureClass:
    """Has methods with wrong names."""

    def process(self, path, query_params, handler):
        return None


class _ReturningHandler:
    """A handler that returns actual HandlerResult dicts."""

    def handle(self, path, query_params, handler):
        return {"body": b'{"ok":true}', "status": 200, "content_type": "application/json"}

    def handle_post(self, path, query_params, handler):
        return {"body": b'{"created":true}', "status": 201}

    def handle_delete(self, path, query_params, handler):
        return {"body": b"", "status": 204}

    def handle_patch(self, path, query_params, handler):
        return {"body": b'{"updated":true}', "status": 200}

    def handle_put(self, path, query_params, handler):
        return {"body": b'{"replaced":true}', "status": 200}


class _AuthenticatedHandler:
    """Implements AuthenticatedHandlerInterface."""

    def get_current_user(self, handler):
        return {"user_id": "test-user"}

    def require_auth_or_error(self, handler):
        return {"user_id": "test-user"}, None


class _UnauthenticatedHandler:
    """AuthenticatedHandler that returns no user."""

    def get_current_user(self, handler):
        return None

    def require_auth_or_error(self, handler):
        return None, {"body": b'{"error":"unauthorized"}', "status": 401}


class _PaginatedHandler:
    """Implements PaginatedHandlerInterface."""

    def get_pagination(self, query_params, default_limit=None, max_limit=None):
        limit = int(query_params.get("limit", default_limit or 20))
        offset = int(query_params.get("offset", 0))
        if max_limit and limit > max_limit:
            limit = max_limit
        return (limit, offset)

    def paginated_response(self, items, total, limit, offset, items_key="items"):
        return {
            "body": b"{}",
            "status": 200,
            "content_type": "application/json",
        }


class _CachedHandler:
    """Implements CachedHandlerInterface."""

    def cached_response(self, cache_key, ttl_seconds, generator):
        return generator()


class _StorageHandler:
    """Implements StorageAccessInterface."""

    def get_storage(self):
        return {"type": "postgres"}

    def get_elo_system(self):
        return {"type": "elo"}


class _PartialAuthHandler:
    """Only has get_current_user, missing require_auth_or_error."""

    def get_current_user(self, handler):
        return None


class _PartialPaginatedHandler:
    """Only has get_pagination, missing paginated_response."""

    def get_pagination(self, query_params, default_limit=None, max_limit=None):
        return (20, 0)


class _PartialStorageHandler:
    """Only has get_storage, missing get_elo_system."""

    def get_storage(self):
        return None


class _MixedHandler:
    """Implements both HandlerInterface and AuthenticatedHandlerInterface."""

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

    def get_current_user(self, handler):
        return None

    def require_auth_or_error(self, handler):
        return None, None


# =============================================================================
# HandlerResult TypedDict Tests
# =============================================================================


class TestHandlerResult:
    """Tests for HandlerResult TypedDict."""

    def test_create_empty(self):
        """HandlerResult is total=False so all fields are optional."""
        result: HandlerResult = {}
        assert result == {}

    def test_create_with_body(self):
        result: HandlerResult = {"body": b'{"ok":true}'}
        assert result["body"] == b'{"ok":true}'

    def test_create_with_status(self):
        result: HandlerResult = {"status": 200}
        assert result["status"] == 200

    def test_create_with_content_type(self):
        result: HandlerResult = {"content_type": "application/json"}
        assert result["content_type"] == "application/json"

    def test_create_with_headers(self):
        result: HandlerResult = {"headers": {"X-Custom": "value"}}
        assert result["headers"]["X-Custom"] == "value"

    def test_create_full(self):
        result: HandlerResult = {
            "body": b'{"data":"test"}',
            "content_type": "application/json",
            "status": 201,
            "headers": {"Location": "/resource/1"},
        }
        assert result["status"] == 201
        assert result["headers"]["Location"] == "/resource/1"
        assert len(result) == 4

    def test_annotations_contain_4_fields(self):
        hints = get_type_hints(HandlerResult)
        assert "body" in hints
        assert "content_type" in hints
        assert "status" in hints
        assert "headers" in hints
        assert len(hints) == 4

    def test_body_type_is_bytes(self):
        hints = get_type_hints(HandlerResult)
        assert hints["body"] is bytes

    def test_status_type_is_int(self):
        hints = get_type_hints(HandlerResult)
        assert hints["status"] is int

    def test_content_type_is_str(self):
        hints = get_type_hints(HandlerResult)
        assert hints["content_type"] is str

    def test_empty_body(self):
        result: HandlerResult = {"body": b"", "status": 204}
        assert result["body"] == b""

    def test_binary_body(self):
        result: HandlerResult = {"body": b"\x00\x01\x02\xff", "status": 200}
        assert len(result["body"]) == 4

    def test_multiple_headers(self):
        result: HandlerResult = {
            "headers": {
                "X-Request-Id": "abc",
                "X-Trace-Id": "xyz",
                "Cache-Control": "no-cache",
            }
        }
        assert len(result["headers"]) == 3

    def test_various_status_codes(self):
        for code in [200, 201, 204, 400, 401, 403, 404, 409, 422, 429, 500, 503]:
            result: HandlerResult = {"status": code}
            assert result["status"] == code


# =============================================================================
# MaybeAsyncHandlerResult Type Alias Tests
# =============================================================================


class TestMaybeAsyncHandlerResult:
    """Tests for MaybeAsyncHandlerResult type alias."""

    def test_alias_is_accessible(self):
        assert MaybeAsyncHandlerResult is not None

    def test_none_is_valid(self):
        """None is a valid MaybeAsyncHandlerResult value."""
        val: MaybeAsyncHandlerResult = None
        assert val is None

    def test_handler_result_is_valid(self):
        """A HandlerResult dict is a valid MaybeAsyncHandlerResult."""
        val: MaybeAsyncHandlerResult = {"body": b"ok", "status": 200}
        assert val["status"] == 200

    def test_awaitable_is_valid(self):
        """An Awaitable[HandlerResult | None] is valid."""

        async def _gen() -> HandlerResult | None:
            return {"body": b"ok", "status": 200}

        val: MaybeAsyncHandlerResult = _gen()
        # It's an awaitable; clean up to avoid RuntimeWarning
        val.close()


# =============================================================================
# HandlerInterface Protocol Tests
# =============================================================================


class TestHandlerInterface:
    """Tests for HandlerInterface runtime_checkable Protocol."""

    def test_is_runtime_checkable(self):
        """HandlerInterface is a runtime_checkable protocol."""
        assert hasattr(HandlerInterface, "_is_runtime_protocol") or hasattr(
            HandlerInterface, "__protocol_attrs__"
        )

    def test_is_protocol_class(self):
        assert issubclass(type(HandlerInterface), type(Protocol))

    def test_complete_handler_satisfies_protocol(self):
        assert isinstance(_CompleteHandler(), HandlerInterface)

    def test_async_complete_handler_satisfies_protocol(self):
        assert isinstance(_AsyncCompleteHandler(), HandlerInterface)

    def test_returning_handler_satisfies_protocol(self):
        assert isinstance(_ReturningHandler(), HandlerInterface)

    def test_handle_only_does_not_satisfy(self):
        assert not isinstance(_HandleOnlyHandler(), HandlerInterface)

    def test_missing_put_does_not_satisfy(self):
        assert not isinstance(_MissingPutHandler(), HandlerInterface)

    def test_missing_delete_does_not_satisfy(self):
        assert not isinstance(_MissingDeleteHandler(), HandlerInterface)

    def test_missing_patch_does_not_satisfy(self):
        assert not isinstance(_MissingPatchHandler(), HandlerInterface)

    def test_missing_post_does_not_satisfy(self):
        assert not isinstance(_MissingPostHandler(), HandlerInterface)

    def test_missing_handle_does_not_satisfy(self):
        assert not isinstance(_MissingHandleMethod(), HandlerInterface)

    def test_empty_class_does_not_satisfy(self):
        assert not isinstance(_EmptyClass(), HandlerInterface)

    def test_none_does_not_satisfy(self):
        assert not isinstance(None, HandlerInterface)

    def test_string_does_not_satisfy(self):
        assert not isinstance("handler", HandlerInterface)

    def test_int_does_not_satisfy(self):
        assert not isinstance(42, HandlerInterface)

    def test_dict_does_not_satisfy(self):
        assert not isinstance({}, HandlerInterface)

    def test_list_does_not_satisfy(self):
        assert not isinstance([], HandlerInterface)

    def test_wrong_signature_class_does_not_satisfy(self):
        assert not isinstance(_WrongSignatureClass(), HandlerInterface)

    def test_mock_with_all_methods_satisfies(self):
        m = MagicMock(spec=_CompleteHandler)
        assert isinstance(m, HandlerInterface)

    def test_protocol_has_handle_method(self):
        assert hasattr(HandlerInterface, "handle")

    def test_protocol_has_handle_post_method(self):
        assert hasattr(HandlerInterface, "handle_post")

    def test_protocol_has_handle_delete_method(self):
        assert hasattr(HandlerInterface, "handle_delete")

    def test_protocol_has_handle_patch_method(self):
        assert hasattr(HandlerInterface, "handle_patch")

    def test_protocol_has_handle_put_method(self):
        assert hasattr(HandlerInterface, "handle_put")

    def test_mixed_handler_satisfies(self):
        """A class with extra methods still satisfies HandlerInterface."""
        assert isinstance(_MixedHandler(), HandlerInterface)

    def test_complete_handler_handle_returns_none(self):
        h = _CompleteHandler()
        assert h.handle("/test", {}, MagicMock()) is None

    def test_complete_handler_handle_post_returns_none(self):
        h = _CompleteHandler()
        assert h.handle_post("/test", {}, MagicMock()) is None

    def test_complete_handler_handle_delete_returns_none(self):
        h = _CompleteHandler()
        assert h.handle_delete("/test", {}, MagicMock()) is None

    def test_complete_handler_handle_patch_returns_none(self):
        h = _CompleteHandler()
        assert h.handle_patch("/test", {}, MagicMock()) is None

    def test_complete_handler_handle_put_returns_none(self):
        h = _CompleteHandler()
        assert h.handle_put("/test", {}, MagicMock()) is None

    def test_returning_handler_handle_returns_result(self):
        h = _ReturningHandler()
        result = h.handle("/test", {}, MagicMock())
        assert result["status"] == 200

    def test_returning_handler_handle_post_returns_result(self):
        h = _ReturningHandler()
        result = h.handle_post("/test", {}, MagicMock())
        assert result["status"] == 201

    def test_returning_handler_handle_delete_returns_result(self):
        h = _ReturningHandler()
        result = h.handle_delete("/test", {}, MagicMock())
        assert result["status"] == 204


# =============================================================================
# AuthenticatedHandlerInterface Protocol Tests
# =============================================================================


class TestAuthenticatedHandlerInterface:
    """Tests for AuthenticatedHandlerInterface runtime_checkable Protocol."""

    def test_is_runtime_checkable(self):
        assert hasattr(AuthenticatedHandlerInterface, "_is_runtime_protocol") or hasattr(
            AuthenticatedHandlerInterface, "__protocol_attrs__"
        )

    def test_authenticated_handler_satisfies(self):
        assert isinstance(_AuthenticatedHandler(), AuthenticatedHandlerInterface)

    def test_unauthenticated_handler_satisfies(self):
        """The protocol only checks method existence, not behavior."""
        assert isinstance(_UnauthenticatedHandler(), AuthenticatedHandlerInterface)

    def test_partial_auth_handler_does_not_satisfy(self):
        """Missing require_auth_or_error should fail."""
        assert not isinstance(_PartialAuthHandler(), AuthenticatedHandlerInterface)

    def test_empty_class_does_not_satisfy(self):
        assert not isinstance(_EmptyClass(), AuthenticatedHandlerInterface)

    def test_none_does_not_satisfy(self):
        assert not isinstance(None, AuthenticatedHandlerInterface)

    def test_complete_handler_does_not_satisfy(self):
        """A handler without auth methods should not satisfy this protocol."""
        assert not isinstance(_CompleteHandler(), AuthenticatedHandlerInterface)

    def test_mixed_handler_satisfies(self):
        """MixedHandler has both handler and auth methods."""
        assert isinstance(_MixedHandler(), AuthenticatedHandlerInterface)

    def test_protocol_has_get_current_user(self):
        assert hasattr(AuthenticatedHandlerInterface, "get_current_user")

    def test_protocol_has_require_auth_or_error(self):
        assert hasattr(AuthenticatedHandlerInterface, "require_auth_or_error")

    def test_authenticated_handler_returns_user(self):
        h = _AuthenticatedHandler()
        user = h.get_current_user(MagicMock())
        assert user is not None
        assert user["user_id"] == "test-user"

    def test_authenticated_handler_require_auth_success(self):
        h = _AuthenticatedHandler()
        user, error = h.require_auth_or_error(MagicMock())
        assert user is not None
        assert error is None

    def test_unauthenticated_handler_returns_none_user(self):
        h = _UnauthenticatedHandler()
        user = h.get_current_user(MagicMock())
        assert user is None

    def test_unauthenticated_handler_require_auth_returns_error(self):
        h = _UnauthenticatedHandler()
        user, error = h.require_auth_or_error(MagicMock())
        assert user is None
        assert error is not None
        assert error["status"] == 401

    def test_mock_with_auth_methods_satisfies(self):
        m = MagicMock(spec=_AuthenticatedHandler)
        assert isinstance(m, AuthenticatedHandlerInterface)


# =============================================================================
# PaginatedHandlerInterface Protocol Tests
# =============================================================================


class TestPaginatedHandlerInterface:
    """Tests for PaginatedHandlerInterface runtime_checkable Protocol."""

    def test_is_runtime_checkable(self):
        assert hasattr(PaginatedHandlerInterface, "_is_runtime_protocol") or hasattr(
            PaginatedHandlerInterface, "__protocol_attrs__"
        )

    def test_paginated_handler_satisfies(self):
        assert isinstance(_PaginatedHandler(), PaginatedHandlerInterface)

    def test_partial_paginated_does_not_satisfy(self):
        assert not isinstance(_PartialPaginatedHandler(), PaginatedHandlerInterface)

    def test_empty_class_does_not_satisfy(self):
        assert not isinstance(_EmptyClass(), PaginatedHandlerInterface)

    def test_none_does_not_satisfy(self):
        assert not isinstance(None, PaginatedHandlerInterface)

    def test_complete_handler_does_not_satisfy(self):
        assert not isinstance(_CompleteHandler(), PaginatedHandlerInterface)

    def test_protocol_has_get_pagination(self):
        assert hasattr(PaginatedHandlerInterface, "get_pagination")

    def test_protocol_has_paginated_response(self):
        assert hasattr(PaginatedHandlerInterface, "paginated_response")

    def test_paginated_handler_get_pagination_defaults(self):
        h = _PaginatedHandler()
        limit, offset = h.get_pagination({})
        assert limit == 20
        assert offset == 0

    def test_paginated_handler_get_pagination_custom(self):
        h = _PaginatedHandler()
        limit, offset = h.get_pagination({"limit": "50", "offset": "10"})
        assert limit == 50
        assert offset == 10

    def test_paginated_handler_max_limit_enforcement(self):
        h = _PaginatedHandler()
        limit, offset = h.get_pagination({"limit": "200"}, max_limit=100)
        assert limit == 100

    def test_paginated_handler_paginated_response_returns_result(self):
        h = _PaginatedHandler()
        result = h.paginated_response([1, 2, 3], total=100, limit=10, offset=0)
        assert result["status"] == 200


# =============================================================================
# CachedHandlerInterface Protocol Tests
# =============================================================================


class TestCachedHandlerInterface:
    """Tests for CachedHandlerInterface runtime_checkable Protocol."""

    def test_is_runtime_checkable(self):
        assert hasattr(CachedHandlerInterface, "_is_runtime_protocol") or hasattr(
            CachedHandlerInterface, "__protocol_attrs__"
        )

    def test_cached_handler_satisfies(self):
        assert isinstance(_CachedHandler(), CachedHandlerInterface)

    def test_empty_class_does_not_satisfy(self):
        assert not isinstance(_EmptyClass(), CachedHandlerInterface)

    def test_none_does_not_satisfy(self):
        assert not isinstance(None, CachedHandlerInterface)

    def test_complete_handler_does_not_satisfy(self):
        assert not isinstance(_CompleteHandler(), CachedHandlerInterface)

    def test_protocol_has_cached_response(self):
        assert hasattr(CachedHandlerInterface, "cached_response")

    def test_cached_handler_invokes_generator(self):
        h = _CachedHandler()
        result = h.cached_response("key", 300, lambda: {"data": "fresh"})
        assert result == {"data": "fresh"}

    def test_cached_handler_with_different_ttls(self):
        h = _CachedHandler()
        for ttl in [0, 1, 60, 300, 3600, 86400]:
            result = h.cached_response(f"key_{ttl}", ttl, lambda: ttl)
            assert result == ttl


# =============================================================================
# StorageAccessInterface Protocol Tests
# =============================================================================


class TestStorageAccessInterface:
    """Tests for StorageAccessInterface Protocol (not runtime_checkable)."""

    def test_is_not_runtime_checkable(self):
        """StorageAccessInterface is a Protocol but NOT runtime_checkable."""
        assert not getattr(StorageAccessInterface, "_is_runtime_protocol", False)

    def test_isinstance_raises_type_error(self):
        """isinstance() raises TypeError for non-runtime_checkable protocols."""
        with pytest.raises(TypeError, match="runtime_checkable"):
            isinstance(_StorageHandler(), StorageAccessInterface)

    def test_is_protocol_class(self):
        assert issubclass(type(StorageAccessInterface), type(Protocol))

    def test_protocol_has_get_storage(self):
        assert hasattr(StorageAccessInterface, "get_storage")

    def test_protocol_has_get_elo_system(self):
        assert hasattr(StorageAccessInterface, "get_elo_system")

    def test_storage_handler_has_required_methods(self):
        """Structural check: _StorageHandler has the required methods."""
        h = _StorageHandler()
        assert hasattr(h, "get_storage")
        assert hasattr(h, "get_elo_system")
        assert callable(h.get_storage)
        assert callable(h.get_elo_system)

    def test_partial_storage_handler_missing_method(self):
        """Structural check: _PartialStorageHandler is missing get_elo_system."""
        h = _PartialStorageHandler()
        assert hasattr(h, "get_storage")
        assert not hasattr(h, "get_elo_system")

    def test_storage_handler_returns_storage(self):
        h = _StorageHandler()
        storage = h.get_storage()
        assert storage is not None
        assert storage["type"] == "postgres"

    def test_storage_handler_returns_elo_system(self):
        h = _StorageHandler()
        elo = h.get_elo_system()
        assert elo is not None
        assert elo["type"] == "elo"


# =============================================================================
# MinimalServerContext TypedDict Tests
# =============================================================================


class TestMinimalServerContext:
    """Tests for MinimalServerContext TypedDict."""

    def test_create_empty(self):
        """MinimalServerContext is total=False so all fields are optional."""
        ctx: MinimalServerContext = {}
        assert ctx == {}

    def test_create_with_storage(self):
        ctx: MinimalServerContext = {"storage": MagicMock()}
        assert "storage" in ctx

    def test_create_with_user_store(self):
        ctx: MinimalServerContext = {"user_store": MagicMock()}
        assert "user_store" in ctx

    def test_create_with_elo_system(self):
        ctx: MinimalServerContext = {"elo_system": MagicMock()}
        assert "elo_system" in ctx

    def test_create_full(self):
        ctx: MinimalServerContext = {
            "storage": MagicMock(),
            "user_store": MagicMock(),
            "elo_system": MagicMock(),
        }
        assert len(ctx) == 3

    def test_annotations_contain_3_fields(self):
        hints = get_type_hints(MinimalServerContext)
        assert "storage" in hints
        assert "user_store" in hints
        assert "elo_system" in hints
        assert len(hints) == 3

    def test_get_safely_with_defaults(self):
        """Handlers should use ctx.get() to safely access optional fields."""
        ctx: MinimalServerContext = {"storage": "db_instance"}
        assert ctx.get("storage") == "db_instance"
        assert ctx.get("user_store") is None
        assert ctx.get("elo_system") is None


# =============================================================================
# RouteConfig TypedDict Tests
# =============================================================================


class TestRouteConfig:
    """Tests for RouteConfig TypedDict."""

    def test_create_empty(self):
        """RouteConfig is total=False so all fields are optional."""
        config: RouteConfig = {}
        assert config == {}

    def test_create_with_path_pattern(self):
        config: RouteConfig = {"path_pattern": "/api/v1/debates/{id}"}
        assert config["path_pattern"] == "/api/v1/debates/{id}"

    def test_create_with_methods(self):
        config: RouteConfig = {"methods": ["GET", "POST"]}
        assert "GET" in config["methods"]
        assert "POST" in config["methods"]

    def test_create_with_handler_class(self):
        config: RouteConfig = {"handler_class": _CompleteHandler}
        assert config["handler_class"] is _CompleteHandler

    def test_create_with_auth_required(self):
        config: RouteConfig = {"requires_auth": True}
        assert config["requires_auth"] is True

    def test_create_with_rate_limit(self):
        config: RouteConfig = {"rate_limit": 100}
        assert config["rate_limit"] == 100

    def test_create_with_no_rate_limit(self):
        config: RouteConfig = {"rate_limit": None}
        assert config["rate_limit"] is None

    def test_create_full(self):
        config: RouteConfig = {
            "path_pattern": "/api/v1/agents",
            "methods": ["GET", "POST", "DELETE"],
            "handler_class": _CompleteHandler,
            "requires_auth": True,
            "rate_limit": 50,
        }
        assert len(config) == 5

    def test_annotations_contain_5_fields(self):
        hints = get_type_hints(RouteConfig)
        assert "path_pattern" in hints
        assert "methods" in hints
        assert "handler_class" in hints
        assert "requires_auth" in hints
        assert "rate_limit" in hints
        assert len(hints) == 5


# =============================================================================
# HandlerRegistration TypedDict Tests
# =============================================================================


class TestHandlerRegistration:
    """Tests for HandlerRegistration TypedDict."""

    def test_create_minimal(self):
        reg: HandlerRegistration = {
            "handler_class": _CompleteHandler,
            "routes": [],
            "lazy": False,
        }
        assert reg["handler_class"] is _CompleteHandler
        assert reg["routes"] == []
        assert reg["lazy"] is False

    def test_create_with_routes(self):
        reg: HandlerRegistration = {
            "handler_class": _CompleteHandler,
            "routes": [
                {"path_pattern": "/api/v1/test", "methods": ["GET"]},
                {"path_pattern": "/api/v1/test/{id}", "methods": ["GET", "PUT", "DELETE"]},
            ],
            "lazy": True,
        }
        assert len(reg["routes"]) == 2
        assert reg["lazy"] is True

    def test_lazy_true(self):
        reg: HandlerRegistration = {
            "handler_class": _EmptyClass,
            "routes": [],
            "lazy": True,
        }
        assert reg["lazy"] is True

    def test_lazy_false(self):
        reg: HandlerRegistration = {
            "handler_class": _EmptyClass,
            "routes": [],
            "lazy": False,
        }
        assert reg["lazy"] is False

    def test_annotations_contain_3_fields(self):
        hints = get_type_hints(HandlerRegistration)
        assert "handler_class" in hints
        assert "routes" in hints
        assert "lazy" in hints
        assert len(hints) == 3


# =============================================================================
# is_handler() Factory Function Tests
# =============================================================================


class TestIsHandler:
    """Tests for the is_handler() factory function."""

    def test_complete_handler_returns_true(self):
        assert is_handler(_CompleteHandler()) is True

    def test_async_handler_returns_true(self):
        assert is_handler(_AsyncCompleteHandler()) is True

    def test_returning_handler_returns_true(self):
        assert is_handler(_ReturningHandler()) is True

    def test_mixed_handler_returns_true(self):
        assert is_handler(_MixedHandler()) is True

    def test_handle_only_returns_false(self):
        assert is_handler(_HandleOnlyHandler()) is False

    def test_missing_put_returns_false(self):
        assert is_handler(_MissingPutHandler()) is False

    def test_missing_delete_returns_false(self):
        assert is_handler(_MissingDeleteHandler()) is False

    def test_missing_patch_returns_false(self):
        assert is_handler(_MissingPatchHandler()) is False

    def test_missing_post_returns_false(self):
        assert is_handler(_MissingPostHandler()) is False

    def test_missing_handle_returns_false(self):
        assert is_handler(_MissingHandleMethod()) is False

    def test_empty_class_returns_false(self):
        assert is_handler(_EmptyClass()) is False

    def test_none_returns_false(self):
        assert is_handler(None) is False

    def test_string_returns_false(self):
        assert is_handler("not a handler") is False

    def test_int_returns_false(self):
        assert is_handler(42) is False

    def test_dict_returns_false(self):
        assert is_handler({}) is False

    def test_list_returns_false(self):
        assert is_handler([]) is False

    def test_wrong_signature_returns_false(self):
        assert is_handler(_WrongSignatureClass()) is False

    def test_mock_with_spec_returns_true(self):
        m = MagicMock(spec=_CompleteHandler)
        assert is_handler(m) is True

    def test_magic_mock_without_spec_returns_true(self):
        """MagicMock without spec has all attribute lookups succeed."""
        m = MagicMock()
        assert is_handler(m) is True


# =============================================================================
# is_authenticated_handler() Factory Function Tests
# =============================================================================


class TestIsAuthenticatedHandler:
    """Tests for the is_authenticated_handler() factory function."""

    def test_authenticated_handler_returns_true(self):
        assert is_authenticated_handler(_AuthenticatedHandler()) is True

    def test_unauthenticated_handler_returns_true(self):
        """Protocol checks structure, not behavior."""
        assert is_authenticated_handler(_UnauthenticatedHandler()) is True

    def test_mixed_handler_returns_true(self):
        assert is_authenticated_handler(_MixedHandler()) is True

    def test_partial_auth_returns_false(self):
        assert is_authenticated_handler(_PartialAuthHandler()) is False

    def test_complete_handler_returns_false(self):
        assert is_authenticated_handler(_CompleteHandler()) is False

    def test_empty_class_returns_false(self):
        assert is_authenticated_handler(_EmptyClass()) is False

    def test_none_returns_false(self):
        assert is_authenticated_handler(None) is False

    def test_string_returns_false(self):
        assert is_authenticated_handler("handler") is False

    def test_int_returns_false(self):
        assert is_authenticated_handler(0) is False

    def test_mock_with_spec_returns_true(self):
        m = MagicMock(spec=_AuthenticatedHandler)
        assert is_authenticated_handler(m) is True


# =============================================================================
# __all__ Export Tests
# =============================================================================


class TestAllExports:
    """Test that __all__ is comprehensive and all items are importable."""

    def test_all_exists(self):
        from aragora.server.handlers import interface

        assert hasattr(interface, "__all__")

    def test_all_is_list(self):
        from aragora.server.handlers import interface

        assert isinstance(interface.__all__, list)

    def test_all_items_are_strings(self):
        from aragora.server.handlers import interface

        for name in interface.__all__:
            assert isinstance(name, str)

    def test_all_items_importable(self):
        from aragora.server.handlers import interface

        for name in interface.__all__:
            assert hasattr(interface, name), f"{name!r} in __all__ but not accessible"

    def test_all_has_no_duplicates(self):
        from aragora.server.handlers import interface

        assert len(interface.__all__) == len(set(interface.__all__))

    def test_handler_result_in_all(self):
        from aragora.server.handlers import interface

        assert "HandlerResult" in interface.__all__

    def test_maybe_async_handler_result_in_all(self):
        from aragora.server.handlers import interface

        assert "MaybeAsyncHandlerResult" in interface.__all__

    def test_handler_interface_in_all(self):
        from aragora.server.handlers import interface

        assert "HandlerInterface" in interface.__all__

    def test_authenticated_handler_interface_in_all(self):
        from aragora.server.handlers import interface

        assert "AuthenticatedHandlerInterface" in interface.__all__

    def test_paginated_handler_interface_in_all(self):
        from aragora.server.handlers import interface

        assert "PaginatedHandlerInterface" in interface.__all__

    def test_cached_handler_interface_in_all(self):
        from aragora.server.handlers import interface

        assert "CachedHandlerInterface" in interface.__all__

    def test_storage_access_interface_in_all(self):
        from aragora.server.handlers import interface

        assert "StorageAccessInterface" in interface.__all__

    def test_minimal_server_context_in_all(self):
        from aragora.server.handlers import interface

        assert "MinimalServerContext" in interface.__all__

    def test_route_config_in_all(self):
        from aragora.server.handlers import interface

        assert "RouteConfig" in interface.__all__

    def test_handler_registration_in_all(self):
        from aragora.server.handlers import interface

        assert "HandlerRegistration" in interface.__all__

    def test_is_handler_in_all(self):
        from aragora.server.handlers import interface

        assert "is_handler" in interface.__all__

    def test_is_authenticated_handler_in_all(self):
        from aragora.server.handlers import interface

        assert "is_authenticated_handler" in interface.__all__

    def test_all_count(self):
        from aragora.server.handlers import interface

        assert len(interface.__all__) == 12


# =============================================================================
# Cross-Protocol Tests
# =============================================================================


class TestCrossProtocol:
    """Test combinations of protocol satisfaction."""

    def test_mixed_handler_satisfies_both_handler_and_auth(self):
        h = _MixedHandler()
        assert isinstance(h, HandlerInterface)
        assert isinstance(h, AuthenticatedHandlerInterface)

    def test_complete_handler_only_satisfies_handler_interface(self):
        h = _CompleteHandler()
        assert isinstance(h, HandlerInterface)
        assert not isinstance(h, AuthenticatedHandlerInterface)
        assert not isinstance(h, PaginatedHandlerInterface)
        assert not isinstance(h, CachedHandlerInterface)
        # StorageAccessInterface is not runtime_checkable, so use structural check
        assert not hasattr(h, "get_storage") or not hasattr(h, "get_elo_system")

    def test_authenticated_only_satisfies_auth_interface(self):
        h = _AuthenticatedHandler()
        assert not isinstance(h, HandlerInterface)
        assert isinstance(h, AuthenticatedHandlerInterface)

    def test_paginated_only_satisfies_paginated_interface(self):
        h = _PaginatedHandler()
        assert not isinstance(h, HandlerInterface)
        assert isinstance(h, PaginatedHandlerInterface)

    def test_cached_only_satisfies_cached_interface(self):
        h = _CachedHandler()
        assert not isinstance(h, HandlerInterface)
        assert isinstance(h, CachedHandlerInterface)

    def test_storage_only_satisfies_storage_structurally(self):
        """StorageAccessInterface is not runtime_checkable, use structural check."""
        h = _StorageHandler()
        assert not isinstance(h, HandlerInterface)
        assert hasattr(h, "get_storage") and hasattr(h, "get_elo_system")

    def test_empty_class_satisfies_none(self):
        h = _EmptyClass()
        assert not isinstance(h, HandlerInterface)
        assert not isinstance(h, AuthenticatedHandlerInterface)
        assert not isinstance(h, PaginatedHandlerInterface)
        assert not isinstance(h, CachedHandlerInterface)
        # StorageAccessInterface is not runtime_checkable
        assert not hasattr(h, "get_storage")


# =============================================================================
# Protocol Method Signature Introspection
# =============================================================================


class TestProtocolMethodSignatures:
    """Test method signatures on protocol classes."""

    def test_handler_interface_handle_params(self):
        sig = inspect.signature(HandlerInterface.handle)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "path" in params
        assert "query_params" in params
        assert "handler" in params

    def test_handler_interface_handle_post_params(self):
        sig = inspect.signature(HandlerInterface.handle_post)
        params = list(sig.parameters.keys())
        assert "path" in params
        assert "query_params" in params
        assert "handler" in params

    def test_handler_interface_handle_delete_params(self):
        sig = inspect.signature(HandlerInterface.handle_delete)
        params = list(sig.parameters.keys())
        assert "path" in params
        assert "query_params" in params
        assert "handler" in params

    def test_handler_interface_handle_patch_params(self):
        sig = inspect.signature(HandlerInterface.handle_patch)
        params = list(sig.parameters.keys())
        assert "path" in params
        assert "query_params" in params
        assert "handler" in params

    def test_handler_interface_handle_put_params(self):
        sig = inspect.signature(HandlerInterface.handle_put)
        params = list(sig.parameters.keys())
        assert "path" in params
        assert "query_params" in params
        assert "handler" in params

    def test_authenticated_get_current_user_params(self):
        sig = inspect.signature(AuthenticatedHandlerInterface.get_current_user)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "handler" in params

    def test_authenticated_require_auth_or_error_params(self):
        sig = inspect.signature(AuthenticatedHandlerInterface.require_auth_or_error)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "handler" in params

    def test_paginated_get_pagination_params(self):
        sig = inspect.signature(PaginatedHandlerInterface.get_pagination)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "query_params" in params
        assert "default_limit" in params
        assert "max_limit" in params

    def test_paginated_paginated_response_params(self):
        sig = inspect.signature(PaginatedHandlerInterface.paginated_response)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "items" in params
        assert "total" in params
        assert "limit" in params
        assert "offset" in params
        assert "items_key" in params

    def test_cached_response_params(self):
        sig = inspect.signature(CachedHandlerInterface.cached_response)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "cache_key" in params
        assert "ttl_seconds" in params
        assert "generator" in params

    def test_storage_get_storage_params(self):
        sig = inspect.signature(StorageAccessInterface.get_storage)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert len(params) == 1  # Only self

    def test_storage_get_elo_system_params(self):
        sig = inspect.signature(StorageAccessInterface.get_elo_system)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert len(params) == 1


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_handler_result_with_unicode_body(self):
        result: HandlerResult = {"body": "unicode: \u2192 \u2714".encode("utf-8")}
        assert b"\xe2" in result["body"]

    def test_handler_result_with_large_body(self):
        result: HandlerResult = {"body": b"x" * 1_000_000, "status": 200}
        assert len(result["body"]) == 1_000_000

    def test_handler_result_with_empty_headers(self):
        result: HandlerResult = {"headers": {}}
        assert result["headers"] == {}

    def test_route_config_with_empty_methods(self):
        config: RouteConfig = {"methods": []}
        assert config["methods"] == []

    def test_route_config_with_all_http_methods(self):
        config: RouteConfig = {
            "methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
        }
        assert len(config["methods"]) == 7

    def test_handler_registration_with_many_routes(self):
        routes = [{"path_pattern": f"/api/v1/resource{i}"} for i in range(50)]
        reg: HandlerRegistration = {
            "handler_class": _CompleteHandler,
            "routes": routes,
            "lazy": True,
        }
        assert len(reg["routes"]) == 50

    def test_is_handler_with_lambda(self):
        """Lambdas are not handlers."""
        assert is_handler(lambda: None) is False

    def test_is_handler_with_function(self):
        """Regular functions are not handlers."""

        def my_func():
            pass

        assert is_handler(my_func) is False

    def test_dynamically_created_handler(self):
        """Dynamically created class satisfying the protocol."""
        DynHandler = type(
            "DynHandler",
            (),
            {
                "handle": lambda self, p, q, h: None,
                "handle_post": lambda self, p, q, h: None,
                "handle_delete": lambda self, p, q, h: None,
                "handle_patch": lambda self, p, q, h: None,
                "handle_put": lambda self, p, q, h: None,
            },
        )
        assert is_handler(DynHandler()) is True

    def test_dynamically_created_incomplete_handler(self):
        """Dynamically created class missing a method."""
        DynHandler = type(
            "DynHandler",
            (),
            {
                "handle": lambda self, p, q, h: None,
                "handle_post": lambda self, p, q, h: None,
            },
        )
        assert is_handler(DynHandler()) is False

    def test_subclassed_handler(self):
        """Subclass of a complete handler still satisfies protocol."""

        class SubHandler(_CompleteHandler):
            def handle(self, path, query_params, handler):
                return {"body": b"sub", "status": 200}

        assert is_handler(SubHandler()) is True

    def test_multiple_inheritance_handler(self):
        """Class inheriting from multiple helper classes."""

        class MultiHandler(_CompleteHandler, _AuthenticatedHandler):
            pass

        h = MultiHandler()
        assert is_handler(h) is True
        assert is_authenticated_handler(h) is True
