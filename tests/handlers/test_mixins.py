"""Tests for handler mixin classes (aragora/server/handlers/mixins.py).

Covers all three mixin classes and their methods:
- PaginatedHandlerMixin:
  - get_pagination(): default values, custom overrides, clamping, edge cases
  - paginated_response(): response format, has_more logic, custom items_key
- CachedHandlerMixin:
  - cached_response(): cache miss, cache hit, TTL expiry, generator invocation
  - async_cached_response(): async cache miss, hit, TTL expiry
- AuthenticatedHandlerMixin:
  - require_auth(): delegation to require_auth_or_error, fallback path, unauthenticated
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.mixins import (
    AuthenticatedHandlerMixin,
    CachedHandlerMixin,
    PaginatedHandlerMixin,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: HandlerResult) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    return json.loads(result.body)


def _status(result: HandlerResult) -> int:
    """Extract HTTP status code from a HandlerResult."""
    return result.status_code


# ---------------------------------------------------------------------------
# Concrete classes for testing (mixins need to be mixed into a class)
# ---------------------------------------------------------------------------


class PaginatedHandler(PaginatedHandlerMixin):
    """Concrete class using PaginatedHandlerMixin."""
    pass


class CachedHandler(CachedHandlerMixin):
    """Concrete class using CachedHandlerMixin."""
    pass


class AuthHandler(AuthenticatedHandlerMixin):
    """Concrete class using AuthenticatedHandlerMixin (no BaseHandler)."""
    pass


class AuthHandlerWithRequireAuth(AuthenticatedHandlerMixin):
    """Concrete class with require_auth_or_error method (simulates BaseHandler)."""

    def __init__(self, user=None, error=None):
        self._user = user
        self._error = error

    def require_auth_or_error(self, handler):
        return self._user, self._error


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def paginated():
    """Create PaginatedHandler instance."""
    return PaginatedHandler()


@pytest.fixture
def cached():
    """Create CachedHandler instance."""
    return CachedHandler()


@pytest.fixture(autouse=True)
def clear_handler_cache():
    """Clear the global handler cache before each test to avoid cross-test pollution."""
    from aragora.server.handlers.admin.cache import clear_cache
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler."""
    handler = MagicMock()
    handler.headers = {}
    handler.client_address = ("127.0.0.1", 12345)
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = b"{}"
    return handler


# ===========================================================================
# PaginatedHandlerMixin Tests
# ===========================================================================


class TestGetPagination:
    """Tests for PaginatedHandlerMixin.get_pagination()."""

    def test_defaults_with_empty_params(self, paginated):
        """Default limit and offset when no params provided."""
        limit, offset = paginated.get_pagination({})
        assert limit == 20
        assert offset == 0

    def test_explicit_limit_and_offset(self, paginated):
        """Explicit limit and offset from query params."""
        limit, offset = paginated.get_pagination({"limit": "10", "offset": "5"})
        assert limit == 10
        assert offset == 5

    def test_limit_as_integer(self, paginated):
        """Integer values in params (not just strings)."""
        limit, offset = paginated.get_pagination({"limit": 15, "offset": 3})
        assert limit == 15
        assert offset == 3

    def test_custom_default_limit(self, paginated):
        """Override default_limit."""
        limit, offset = paginated.get_pagination({}, default_limit=50)
        assert limit == 50
        assert offset == 0

    def test_custom_max_limit(self, paginated):
        """Override max_limit."""
        limit, offset = paginated.get_pagination({"limit": "200"}, max_limit=150)
        assert limit == 150
        assert offset == 0

    def test_limit_clamped_to_max(self, paginated):
        """Limit exceeding MAX_LIMIT gets clamped."""
        limit, offset = paginated.get_pagination({"limit": "500"})
        assert limit == 100  # MAX_LIMIT default
        assert offset == 0

    def test_limit_clamped_to_min_one(self, paginated):
        """Limit below 1 gets clamped to 1."""
        limit, offset = paginated.get_pagination({"limit": "0"})
        assert limit == 1

    def test_negative_limit(self, paginated):
        """Negative limit gets clamped to 1."""
        limit, offset = paginated.get_pagination({"limit": "-5"})
        assert limit == 1

    def test_negative_offset_clamped_to_zero(self, paginated):
        """Negative offset gets clamped to 0."""
        limit, offset = paginated.get_pagination({"offset": "-10"})
        assert offset == 0

    def test_zero_offset(self, paginated):
        """Zero offset is valid."""
        limit, offset = paginated.get_pagination({"offset": "0"})
        assert offset == 0

    def test_large_offset(self, paginated):
        """Large offset is allowed."""
        limit, offset = paginated.get_pagination({"offset": "10000"})
        assert offset == 10000

    def test_invalid_limit_uses_default(self, paginated):
        """Non-numeric limit falls back to default."""
        limit, offset = paginated.get_pagination({"limit": "abc"})
        assert limit == 20  # DEFAULT_LIMIT

    def test_invalid_offset_uses_default(self, paginated):
        """Non-numeric offset falls back to 0."""
        limit, offset = paginated.get_pagination({"offset": "xyz"})
        assert offset == 0

    def test_custom_default_and_max_limit(self, paginated):
        """Both default_limit and max_limit overridden."""
        limit, offset = paginated.get_pagination(
            {}, default_limit=30, max_limit=50
        )
        assert limit == 30

    def test_custom_max_limit_exceeded(self, paginated):
        """Explicit limit exceeding custom max_limit gets clamped."""
        limit, offset = paginated.get_pagination(
            {"limit": "80"}, max_limit=50
        )
        assert limit == 50

    def test_limit_exactly_at_max(self, paginated):
        """Limit exactly at MAX_LIMIT is allowed."""
        limit, offset = paginated.get_pagination({"limit": "100"})
        assert limit == 100

    def test_limit_one(self, paginated):
        """Limit of 1 is the minimum valid value."""
        limit, offset = paginated.get_pagination({"limit": "1"})
        assert limit == 1

    def test_class_constants(self):
        """Verify class-level default constants."""
        assert PaginatedHandlerMixin.DEFAULT_LIMIT == 20
        assert PaginatedHandlerMixin.MAX_LIMIT == 100
        assert PaginatedHandlerMixin.DEFAULT_OFFSET == 0

    def test_list_value_for_limit(self, paginated):
        """List values from query strings use first element."""
        limit, offset = paginated.get_pagination({"limit": ["25", "50"]})
        assert limit == 25

    def test_list_value_for_offset(self, paginated):
        """List values from query strings use first element for offset."""
        limit, offset = paginated.get_pagination({"offset": ["10", "20"]})
        assert offset == 10


class TestPaginatedResponse:
    """Tests for PaginatedHandlerMixin.paginated_response()."""

    def test_basic_response(self, paginated):
        """Basic paginated response structure."""
        items = [{"id": 1}, {"id": 2}]
        result = paginated.paginated_response(items, total=10, limit=2, offset=0)
        body = _body(result)
        assert body["items"] == items
        assert body["total"] == 10
        assert body["limit"] == 2
        assert body["offset"] == 0
        assert body["has_more"] is True

    def test_has_more_true(self, paginated):
        """has_more is True when more items exist."""
        result = paginated.paginated_response(
            [1, 2, 3], total=10, limit=3, offset=0
        )
        body = _body(result)
        assert body["has_more"] is True

    def test_has_more_false_last_page(self, paginated):
        """has_more is False on last page."""
        result = paginated.paginated_response(
            [8, 9, 10], total=10, limit=3, offset=7
        )
        body = _body(result)
        assert body["has_more"] is False

    def test_has_more_false_exact_boundary(self, paginated):
        """has_more is False when offset + items == total."""
        result = paginated.paginated_response(
            [1, 2, 3, 4, 5], total=5, limit=5, offset=0
        )
        body = _body(result)
        assert body["has_more"] is False

    def test_has_more_with_offset(self, paginated):
        """has_more with offset correctly calculates remaining items."""
        result = paginated.paginated_response(
            [4, 5], total=10, limit=2, offset=3
        )
        body = _body(result)
        # offset=3, len(items)=2, total=10: 3+2 < 10 -> True
        assert body["has_more"] is True

    def test_empty_items(self, paginated):
        """Empty items list."""
        result = paginated.paginated_response([], total=0, limit=20, offset=0)
        body = _body(result)
        assert body["items"] == []
        assert body["total"] == 0
        assert body["has_more"] is False

    def test_empty_items_with_total(self, paginated):
        """Empty items with non-zero total (e.g., offset past end)."""
        result = paginated.paginated_response([], total=100, limit=20, offset=100)
        body = _body(result)
        assert body["items"] == []
        assert body["total"] == 100
        # offset=100, len([])=0, total=100: 100+0 < 100 -> False
        assert body["has_more"] is False

    def test_custom_items_key(self, paginated):
        """Custom items key instead of 'items'."""
        items = [{"name": "a"}, {"name": "b"}]
        result = paginated.paginated_response(
            items, total=5, limit=2, offset=0, items_key="results"
        )
        body = _body(result)
        assert "results" in body
        assert body["results"] == items
        assert "items" not in body

    def test_response_status_code(self, paginated):
        """Response has 200 status code."""
        result = paginated.paginated_response([1], total=1, limit=10, offset=0)
        assert _status(result) == 200

    def test_response_content_type(self, paginated):
        """Response has JSON content type."""
        result = paginated.paginated_response([1], total=1, limit=10, offset=0)
        assert result.content_type == "application/json"

    def test_single_item(self, paginated):
        """Single item response."""
        result = paginated.paginated_response(
            [{"id": "only"}], total=1, limit=20, offset=0
        )
        body = _body(result)
        assert len(body["items"]) == 1
        assert body["total"] == 1
        assert body["has_more"] is False

    def test_large_total(self, paginated):
        """Large total with small page."""
        result = paginated.paginated_response(
            [1, 2], total=100000, limit=2, offset=0
        )
        body = _body(result)
        assert body["total"] == 100000
        assert body["has_more"] is True


class TestPaginationEndToEnd:
    """End-to-end tests combining get_pagination and paginated_response."""

    def test_full_flow(self, paginated):
        """Simulate a full pagination flow: extract params then build response."""
        limit, offset = paginated.get_pagination({"limit": "5", "offset": "10"})
        all_items = list(range(100))
        page_items = all_items[offset : offset + limit]
        result = paginated.paginated_response(
            page_items, total=len(all_items), limit=limit, offset=offset
        )
        body = _body(result)
        assert body["items"] == [10, 11, 12, 13, 14]
        assert body["limit"] == 5
        assert body["offset"] == 10
        assert body["total"] == 100
        assert body["has_more"] is True

    def test_last_page_flow(self, paginated):
        """Full flow for the last page of results."""
        limit, offset = paginated.get_pagination({"limit": "10", "offset": "95"})
        all_items = list(range(100))
        page_items = all_items[offset : offset + limit]
        result = paginated.paginated_response(
            page_items, total=len(all_items), limit=limit, offset=offset
        )
        body = _body(result)
        assert body["items"] == [95, 96, 97, 98, 99]
        assert body["has_more"] is False


# ===========================================================================
# CachedHandlerMixin Tests
# ===========================================================================


class TestCachedResponse:
    """Tests for CachedHandlerMixin.cached_response()."""

    def test_cache_miss_calls_generator(self, cached):
        """On cache miss, generator is called and value returned."""
        generator = MagicMock(return_value={"data": "fresh"})
        result = cached.cached_response("key1", ttl_seconds=60, generator=generator)
        assert result == {"data": "fresh"}
        generator.assert_called_once()

    def test_cache_hit_returns_cached(self, cached):
        """On cache hit, cached value returned without calling generator."""
        generator1 = MagicMock(return_value="first_value")
        generator2 = MagicMock(return_value="second_value")

        result1 = cached.cached_response("key2", ttl_seconds=60, generator=generator1)
        result2 = cached.cached_response("key2", ttl_seconds=60, generator=generator2)

        assert result1 == "first_value"
        assert result2 == "first_value"
        generator1.assert_called_once()
        generator2.assert_not_called()

    def test_different_keys_independent(self, cached):
        """Different cache keys store independent values."""
        gen_a = MagicMock(return_value="value_a")
        gen_b = MagicMock(return_value="value_b")

        result_a = cached.cached_response("key_a", ttl_seconds=60, generator=gen_a)
        result_b = cached.cached_response("key_b", ttl_seconds=60, generator=gen_b)

        assert result_a == "value_a"
        assert result_b == "value_b"

    def test_ttl_expiry(self, cached):
        """After TTL expires, generator is called again."""
        call_count = 0

        def generator():
            nonlocal call_count
            call_count += 1
            return f"value_{call_count}"

        # First call - cache miss
        result1 = cached.cached_response("ttl_key", ttl_seconds=0.1, generator=generator)
        assert result1 == "value_1"
        assert call_count == 1

        # Immediately - cache hit
        result2 = cached.cached_response("ttl_key", ttl_seconds=0.1, generator=generator)
        assert result2 == "value_1"
        assert call_count == 1

        # Wait for TTL to expire
        time.sleep(0.15)

        # After expiry - cache miss again
        result3 = cached.cached_response("ttl_key", ttl_seconds=0.1, generator=generator)
        assert result3 == "value_2"
        assert call_count == 2

    def test_none_value_cached(self, cached):
        """None return values are cached (not treated as miss)."""
        gen = MagicMock(return_value=None)

        result1 = cached.cached_response("none_key", ttl_seconds=60, generator=gen)
        result2 = cached.cached_response("none_key", ttl_seconds=60, generator=gen)

        assert result1 is None
        assert result2 is None
        gen.assert_called_once()

    def test_empty_list_cached(self, cached):
        """Empty list return values are cached."""
        gen = MagicMock(return_value=[])

        result1 = cached.cached_response("empty_key", ttl_seconds=60, generator=gen)
        result2 = cached.cached_response("empty_key", ttl_seconds=60, generator=gen)

        assert result1 == []
        assert result2 == []
        gen.assert_called_once()

    def test_generator_exception_propagates(self, cached):
        """Generator exceptions propagate to caller."""
        def bad_generator():
            raise ValueError("generator failed")

        with pytest.raises(ValueError, match="generator failed"):
            cached.cached_response("err_key", ttl_seconds=60, generator=bad_generator)

    def test_complex_value_cached(self, cached):
        """Complex nested structures are cached correctly."""
        complex_value = {
            "users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            "meta": {"total": 2, "page": 1},
        }
        gen = MagicMock(return_value=complex_value)

        result = cached.cached_response("complex_key", ttl_seconds=60, generator=gen)
        assert result == complex_value

    def test_zero_ttl_always_misses(self, cached):
        """Zero TTL means every access is a miss."""
        call_count = 0

        def generator():
            nonlocal call_count
            call_count += 1
            return call_count

        # With TTL=0, even consecutive calls should be misses
        # (because time.time() - cached_time >= 0 is always true)
        result1 = cached.cached_response("zero_ttl", ttl_seconds=0, generator=generator)
        # Sleep a tiny bit to ensure time.time() advances
        time.sleep(0.001)
        result2 = cached.cached_response("zero_ttl", ttl_seconds=0, generator=generator)

        assert result1 == 1
        assert result2 == 2
        assert call_count == 2


class TestAsyncCachedResponse:
    """Tests for CachedHandlerMixin.async_cached_response()."""

    @pytest.mark.asyncio
    async def test_async_cache_miss(self, cached):
        """Async cache miss calls generator and returns value."""
        async def generator():
            return {"async": "data"}

        result = await cached.async_cached_response(
            "async_key1", ttl_seconds=60, generator=generator
        )
        assert result == {"async": "data"}

    @pytest.mark.asyncio
    async def test_async_cache_hit(self, cached):
        """Async cache hit returns cached value."""
        call_count = 0

        async def generator():
            nonlocal call_count
            call_count += 1
            return f"value_{call_count}"

        r1 = await cached.async_cached_response("async_hit", ttl_seconds=60, generator=generator)
        r2 = await cached.async_cached_response("async_hit", ttl_seconds=60, generator=generator)

        assert r1 == "value_1"
        assert r2 == "value_1"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_ttl_expiry(self, cached):
        """Async TTL expiry causes re-generation."""
        call_count = 0

        async def generator():
            nonlocal call_count
            call_count += 1
            return call_count

        r1 = await cached.async_cached_response("async_ttl", ttl_seconds=0.1, generator=generator)
        assert r1 == 1

        await asyncio.sleep(0.15)

        r2 = await cached.async_cached_response("async_ttl", ttl_seconds=0.1, generator=generator)
        assert r2 == 2

    @pytest.mark.asyncio
    async def test_async_generator_exception(self, cached):
        """Async generator exceptions propagate."""
        async def bad_generator():
            raise RuntimeError("async fail")

        with pytest.raises(RuntimeError, match="async fail"):
            await cached.async_cached_response(
                "async_err", ttl_seconds=60, generator=bad_generator
            )

    @pytest.mark.asyncio
    async def test_async_none_cached(self, cached):
        """Async None values are cached."""
        call_count = 0

        async def generator():
            nonlocal call_count
            call_count += 1
            return None

        r1 = await cached.async_cached_response("async_none", ttl_seconds=60, generator=generator)
        r2 = await cached.async_cached_response("async_none", ttl_seconds=60, generator=generator)

        assert r1 is None
        assert r2 is None
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_sync_and_async_share_cache(self, cached):
        """Sync and async methods share the same underlying cache."""
        sync_gen = MagicMock(return_value="sync_value")

        # Store via sync
        cached.cached_response("shared_key", ttl_seconds=60, generator=sync_gen)

        # Retrieve via async
        async def async_gen():
            return "async_value"

        result = await cached.async_cached_response(
            "shared_key", ttl_seconds=60, generator=async_gen
        )
        assert result == "sync_value"


# ===========================================================================
# AuthenticatedHandlerMixin Tests
# ===========================================================================


class TestRequireAuthWithBaseHandler:
    """Tests for require_auth when require_auth_or_error is available."""

    def test_authenticated_user_returned(self, mock_http_handler):
        """When user is authenticated, user context is returned."""
        mock_user = MagicMock()
        mock_user.user_id = "user-123"
        handler = AuthHandlerWithRequireAuth(user=mock_user, error=None)

        result = handler.require_auth(mock_http_handler)
        assert result is mock_user

    def test_error_returned_on_auth_failure(self, mock_http_handler):
        """When auth fails, error response is returned."""
        from aragora.server.handlers.utils.responses import error_response

        err = error_response("Auth failed", 401)
        handler = AuthHandlerWithRequireAuth(user=None, error=err)

        result = handler.require_auth(mock_http_handler)
        assert result is err
        assert _status(result) == 401

    def test_error_returned_on_missing_token(self, mock_http_handler):
        """When no token is present, error response from require_auth_or_error is returned."""
        from aragora.server.handlers.utils.responses import error_response

        err = error_response("No token", 401)
        handler = AuthHandlerWithRequireAuth(user=None, error=err)

        result = handler.require_auth(mock_http_handler)
        assert _status(result) == 401


class TestRequireAuthFallback:
    """Tests for require_auth fallback path (no require_auth_or_error)."""

    def test_unauthenticated_returns_401(self, mock_http_handler):
        """Fallback path returns 401 when user is not authenticated."""
        handler = AuthHandler()

        mock_user_ctx = MagicMock()
        mock_user_ctx.is_authenticated = False

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=mock_user_ctx,
        ):
            result = handler.require_auth(mock_http_handler)
            assert isinstance(result, HandlerResult)
            assert _status(result) == 401
            body = _body(result)
            assert "Authentication required" in body["error"]

    def test_authenticated_returns_user_context(self, mock_http_handler):
        """Fallback path returns user context when authenticated."""
        handler = AuthHandler()

        mock_user_ctx = MagicMock()
        mock_user_ctx.is_authenticated = True
        mock_user_ctx.user_id = "user-456"

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=mock_user_ctx,
        ):
            result = handler.require_auth(mock_http_handler)
            assert result is mock_user_ctx
            assert result.user_id == "user-456"

    def test_fallback_uses_ctx_user_store(self, mock_http_handler):
        """Fallback path uses ctx['user_store'] if available."""
        handler = AuthHandler()
        mock_store = MagicMock()
        handler.ctx = {"user_store": mock_store}

        mock_user_ctx = MagicMock()
        mock_user_ctx.is_authenticated = True

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=mock_user_ctx,
        ) as mock_extract:
            handler.require_auth(mock_http_handler)
            mock_extract.assert_called_once_with(mock_http_handler, mock_store)

    def test_fallback_uses_class_user_store(self, mock_http_handler):
        """Fallback path uses class-level user_store if available."""

        class HandlerWithClassStore(AuthenticatedHandlerMixin):
            user_store = MagicMock()

        handler = HandlerWithClassStore()

        mock_user_ctx = MagicMock()
        mock_user_ctx.is_authenticated = True

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=mock_user_ctx,
        ) as mock_extract:
            handler.require_auth(mock_http_handler)
            mock_extract.assert_called_once_with(
                mock_http_handler, HandlerWithClassStore.user_store
            )

    def test_fallback_no_user_store(self, mock_http_handler):
        """Fallback path with no user_store passes None."""
        handler = AuthHandler()

        mock_user_ctx = MagicMock()
        mock_user_ctx.is_authenticated = True

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=mock_user_ctx,
        ) as mock_extract:
            handler.require_auth(mock_http_handler)
            mock_extract.assert_called_once_with(mock_http_handler, None)


# ===========================================================================
# Module-Level Tests
# ===========================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports(self):
        """Module exports all three mixin classes."""
        from aragora.server.handlers import mixins

        assert "PaginatedHandlerMixin" in mixins.__all__
        assert "CachedHandlerMixin" in mixins.__all__
        assert "AuthenticatedHandlerMixin" in mixins.__all__

    def test_all_exports_count(self):
        """Module exports exactly 3 items."""
        from aragora.server.handlers import mixins

        assert len(mixins.__all__) == 3


# ===========================================================================
# Combined Mixin Tests
# ===========================================================================


class TestCombinedMixins:
    """Tests for classes using multiple mixins together."""

    def test_paginated_and_cached_combined(self):
        """A handler class can use both pagination and caching mixins."""

        class CombinedHandler(PaginatedHandlerMixin, CachedHandlerMixin):
            pass

        handler = CombinedHandler()

        # Use pagination
        limit, offset = handler.get_pagination({"limit": "5"})
        assert limit == 5

        # Use caching
        gen = MagicMock(return_value=[1, 2, 3, 4, 5])
        data = handler.cached_response("combo_key", ttl_seconds=60, generator=gen)
        assert data == [1, 2, 3, 4, 5]

        # Build paginated response
        result = handler.paginated_response(data, total=20, limit=limit, offset=offset)
        body = _body(result)
        assert body["items"] == [1, 2, 3, 4, 5]
        assert body["has_more"] is True

    def test_all_three_mixins(self, mock_http_handler):
        """A handler class can combine all three mixins."""

        class FullHandler(
            PaginatedHandlerMixin, CachedHandlerMixin, AuthenticatedHandlerMixin
        ):
            def require_auth_or_error(self, handler):
                mock_user = MagicMock()
                mock_user.user_id = "all-three"
                return mock_user, None

        handler = FullHandler()

        # Auth
        user = handler.require_auth(mock_http_handler)
        assert user.user_id == "all-three"

        # Paginate
        limit, offset = handler.get_pagination({"limit": "3"})
        assert limit == 3

        # Cache
        gen = MagicMock(return_value=["a", "b", "c"])
        data = handler.cached_response("full_key", ttl_seconds=60, generator=gen)

        # Response
        result = handler.paginated_response(data, total=10, limit=limit, offset=offset)
        body = _body(result)
        assert body["items"] == ["a", "b", "c"]


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestPaginationEdgeCases:
    """Edge case tests for pagination."""

    def test_huge_limit_value(self, paginated):
        """Very large limit clamped to MAX_LIMIT."""
        limit, _ = paginated.get_pagination({"limit": "999999999"})
        assert limit == 100

    def test_float_limit_truncated(self, paginated):
        """Float string limit is parsed as int (truncated)."""
        limit, _ = paginated.get_pagination({"limit": "10.5"})
        # get_int_param will fail on "10.5" and return default
        assert limit == 20  # falls back to default

    def test_response_with_mixed_types(self, paginated):
        """Items list can contain mixed types."""
        items = [1, "two", {"three": 3}, None, [5]]
        result = paginated.paginated_response(items, total=5, limit=10, offset=0)
        body = _body(result)
        assert body["items"] == items
        assert body["has_more"] is False


class TestCacheEdgeCases:
    """Edge case tests for caching."""

    def test_empty_cache_key(self, cached):
        """Empty string as cache key works."""
        gen = MagicMock(return_value="empty_key_val")
        result = cached.cached_response("", ttl_seconds=60, generator=gen)
        assert result == "empty_key_val"

    def test_special_chars_in_cache_key(self, cached):
        """Special characters in cache key work."""
        gen = MagicMock(return_value="special")
        result = cached.cached_response(
            "key:with/special\\chars!@#$", ttl_seconds=60, generator=gen
        )
        assert result == "special"

    def test_very_long_cache_key(self, cached):
        """Very long cache key works."""
        long_key = "k" * 10000
        gen = MagicMock(return_value="long_key_val")
        result = cached.cached_response(long_key, ttl_seconds=60, generator=gen)
        assert result == "long_key_val"

    def test_large_cached_value(self, cached):
        """Large values can be cached."""
        big_list = list(range(10000))
        gen = MagicMock(return_value=big_list)
        result = cached.cached_response("big_val", ttl_seconds=60, generator=gen)
        assert result == big_list
        assert len(result) == 10000

    def test_boolean_false_cached(self, cached):
        """False return value is cached (not treated as miss)."""
        gen = MagicMock(return_value=False)

        r1 = cached.cached_response("bool_key", ttl_seconds=60, generator=gen)
        r2 = cached.cached_response("bool_key", ttl_seconds=60, generator=gen)

        assert r1 is False
        assert r2 is False
        gen.assert_called_once()

    def test_zero_value_cached(self, cached):
        """Zero integer return value is cached."""
        gen = MagicMock(return_value=0)

        r1 = cached.cached_response("zero_key", ttl_seconds=60, generator=gen)
        r2 = cached.cached_response("zero_key", ttl_seconds=60, generator=gen)

        assert r1 == 0
        assert r2 == 0
        gen.assert_called_once()

    def test_empty_string_cached(self, cached):
        """Empty string return value is cached."""
        gen = MagicMock(return_value="")

        r1 = cached.cached_response("empty_str_key", ttl_seconds=60, generator=gen)
        r2 = cached.cached_response("empty_str_key", ttl_seconds=60, generator=gen)

        assert r1 == ""
        assert r2 == ""
        gen.assert_called_once()

    def test_negative_ttl_always_misses(self, cached):
        """Negative TTL means every access is a miss (time diff always > negative)."""
        call_count = 0

        def gen():
            nonlocal call_count
            call_count += 1
            return call_count

        r1 = cached.cached_response("neg_ttl", ttl_seconds=-1, generator=gen)
        r2 = cached.cached_response("neg_ttl", ttl_seconds=-1, generator=gen)

        assert r1 == 1
        assert r2 == 2


class TestAuthEdgeCases:
    """Edge case tests for authentication mixin."""

    def test_require_auth_or_error_returns_none_user(self, mock_http_handler):
        """When require_auth_or_error returns (None, error), error is returned."""
        err = MagicMock()
        err.status_code = 403
        handler = AuthHandlerWithRequireAuth(user=None, error=err)

        result = handler.require_auth(mock_http_handler)
        assert result is err

    def test_handler_with_hasattr_require_auth_or_error(self, mock_http_handler):
        """hasattr check correctly detects require_auth_or_error."""
        # AuthHandlerWithRequireAuth has it
        handler = AuthHandlerWithRequireAuth(user=MagicMock(), error=None)
        assert hasattr(handler, "require_auth_or_error")

        # AuthHandler does not have it
        handler2 = AuthHandler()
        assert not hasattr(handler2, "require_auth_or_error")
