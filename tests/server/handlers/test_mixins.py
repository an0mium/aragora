"""
Tests for handler mixin classes: PaginatedHandlerMixin, CachedHandlerMixin,
and AuthenticatedHandlerMixin.

Tests cover:
- PaginatedHandlerMixin: get_pagination(), paginated_response()
- CachedHandlerMixin: cached_response(), async_cached_response()
- AuthenticatedHandlerMixin: require_auth() with various states
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.mixins import (
    AuthenticatedHandlerMixin,
    CachedHandlerMixin,
    PaginatedHandlerMixin,
)


# ===========================================================================
# PaginatedHandlerMixin Tests
# ===========================================================================


class TestPaginatedHandlerMixin:
    """Tests for PaginatedHandlerMixin."""

    def _create_mixin(self):
        return PaginatedHandlerMixin()

    def test_default_class_constants(self):
        mixin = self._create_mixin()
        assert mixin.DEFAULT_LIMIT == 20
        assert mixin.MAX_LIMIT == 100
        assert mixin.DEFAULT_OFFSET == 0

    # ---- get_pagination ----

    def test_get_pagination_defaults(self):
        mixin = self._create_mixin()
        limit, offset = mixin.get_pagination({})
        assert limit == 20
        assert offset == 0

    def test_get_pagination_from_params(self):
        mixin = self._create_mixin()
        limit, offset = mixin.get_pagination({"limit": "50", "offset": "10"})
        assert limit == 50
        assert offset == 10

    def test_get_pagination_clamps_limit_to_max(self):
        mixin = self._create_mixin()
        limit, offset = mixin.get_pagination({"limit": "999"})
        assert limit == 100  # MAX_LIMIT

    def test_get_pagination_clamps_limit_to_min(self):
        mixin = self._create_mixin()
        limit, offset = mixin.get_pagination({"limit": "0"})
        assert limit == 1  # minimum is 1

    def test_get_pagination_negative_limit(self):
        mixin = self._create_mixin()
        limit, offset = mixin.get_pagination({"limit": "-5"})
        assert limit >= 1

    def test_get_pagination_clamps_negative_offset(self):
        mixin = self._create_mixin()
        limit, offset = mixin.get_pagination({"offset": "-10"})
        assert offset == 0

    def test_get_pagination_custom_default_limit(self):
        mixin = self._create_mixin()
        limit, offset = mixin.get_pagination({}, default_limit=5)
        assert limit == 5

    def test_get_pagination_custom_max_limit(self):
        mixin = self._create_mixin()
        limit, offset = mixin.get_pagination({"limit": "200"}, max_limit=50)
        assert limit == 50

    # ---- paginated_response ----

    def test_paginated_response_basic(self):
        mixin = self._create_mixin()
        result = mixin.paginated_response(
            items=["a", "b", "c"],
            total=10,
            limit=3,
            offset=0,
        )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["items"] == ["a", "b", "c"]
        assert body["total"] == 10
        assert body["limit"] == 3
        assert body["offset"] == 0
        assert body["has_more"] is True

    def test_paginated_response_no_more(self):
        mixin = self._create_mixin()
        result = mixin.paginated_response(
            items=["a", "b"],
            total=2,
            limit=10,
            offset=0,
        )
        body = json.loads(result.body)
        assert body["has_more"] is False

    def test_paginated_response_empty(self):
        mixin = self._create_mixin()
        result = mixin.paginated_response(
            items=[],
            total=0,
            limit=20,
            offset=0,
        )
        body = json.loads(result.body)
        assert body["items"] == []
        assert body["total"] == 0
        assert body["has_more"] is False

    def test_paginated_response_custom_items_key(self):
        mixin = self._create_mixin()
        result = mixin.paginated_response(
            items=[1, 2, 3],
            total=3,
            limit=3,
            offset=0,
            items_key="debates",
        )
        body = json.loads(result.body)
        assert "debates" in body
        assert body["debates"] == [1, 2, 3]
        assert "items" not in body

    def test_paginated_response_with_offset(self):
        mixin = self._create_mixin()
        result = mixin.paginated_response(
            items=["c", "d"],
            total=5,
            limit=2,
            offset=2,
        )
        body = json.loads(result.body)
        assert body["offset"] == 2
        assert body["has_more"] is True  # 2 + 2 < 5

    def test_paginated_response_last_page(self):
        mixin = self._create_mixin()
        result = mixin.paginated_response(
            items=["e"],
            total=5,
            limit=2,
            offset=4,
        )
        body = json.loads(result.body)
        assert body["has_more"] is False  # 4 + 1 >= 5


# ===========================================================================
# CachedHandlerMixin Tests
# ===========================================================================


class TestCachedHandlerMixin:
    """Tests for CachedHandlerMixin."""

    def _create_mixin(self):
        return CachedHandlerMixin()

    @patch("aragora.server.handlers.mixins.get_handler_cache")
    def test_cached_response_cache_hit(self, mock_get_cache):
        mock_cache = MagicMock()
        mock_cache.get.return_value = (True, {"cached": True})
        mock_get_cache.return_value = mock_cache

        mixin = self._create_mixin()
        result = mixin.cached_response("key", 300, lambda: {"fresh": True})
        assert result == {"cached": True}
        mock_cache.set.assert_not_called()

    @patch("aragora.server.handlers.mixins.get_handler_cache")
    def test_cached_response_cache_miss(self, mock_get_cache):
        mock_cache = MagicMock()
        mock_cache.get.return_value = (False, None)
        mock_get_cache.return_value = mock_cache

        mixin = self._create_mixin()
        result = mixin.cached_response("key", 300, lambda: {"fresh": True})
        assert result == {"fresh": True}
        mock_cache.set.assert_called_once_with("key", {"fresh": True})

    @patch("aragora.server.handlers.mixins.get_handler_cache")
    def test_cached_response_generator_called_on_miss(self, mock_get_cache):
        mock_cache = MagicMock()
        mock_cache.get.return_value = (False, None)
        mock_get_cache.return_value = mock_cache

        call_count = 0

        def generator():
            nonlocal call_count
            call_count += 1
            return "generated"

        mixin = self._create_mixin()
        mixin.cached_response("key", 60, generator)
        assert call_count == 1

    @patch("aragora.server.handlers.mixins.get_handler_cache")
    def test_cached_response_generator_not_called_on_hit(self, mock_get_cache):
        mock_cache = MagicMock()
        mock_cache.get.return_value = (True, "cached")
        mock_get_cache.return_value = mock_cache

        call_count = 0

        def generator():
            nonlocal call_count
            call_count += 1
            return "generated"

        mixin = self._create_mixin()
        mixin.cached_response("key", 60, generator)
        assert call_count == 0

    # ---- async_cached_response ----

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.mixins.get_handler_cache")
    async def test_async_cached_response_hit(self, mock_get_cache):
        mock_cache = MagicMock()
        mock_cache.get.return_value = (True, "async_cached")
        mock_get_cache.return_value = mock_cache

        mixin = self._create_mixin()
        result = await mixin.async_cached_response("key", 300, AsyncMock(return_value="fresh"))
        assert result == "async_cached"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.mixins.get_handler_cache")
    async def test_async_cached_response_miss(self, mock_get_cache):
        mock_cache = MagicMock()
        mock_cache.get.return_value = (False, None)
        mock_get_cache.return_value = mock_cache

        mixin = self._create_mixin()
        gen = AsyncMock(return_value="async_fresh")
        result = await mixin.async_cached_response("key", 300, gen)
        assert result == "async_fresh"
        mock_cache.set.assert_called_once_with("key", "async_fresh")


# ===========================================================================
# AuthenticatedHandlerMixin Tests
# ===========================================================================


class TestAuthenticatedHandlerMixin:
    """Tests for AuthenticatedHandlerMixin."""

    def test_require_auth_delegates_to_require_auth_or_error(self):
        """When composing class has require_auth_or_error, delegate to it."""

        class Composed(AuthenticatedHandlerMixin):
            def require_auth_or_error(self, handler):
                mock_user = MagicMock()
                mock_user.user_id = "user-42"
                return mock_user, None

        obj = Composed()
        result = obj.require_auth(MagicMock())
        assert result.user_id == "user-42"

    def test_require_auth_returns_error_from_delegate(self):
        """When require_auth_or_error returns an error, propagate it."""

        class Composed(AuthenticatedHandlerMixin):
            def require_auth_or_error(self, handler):
                return None, MagicMock(status_code=401)

        obj = Composed()
        result = obj.require_auth(MagicMock())
        assert result.status_code == 401

    @patch("aragora.server.handlers.mixins.extract_user_from_request")
    def test_require_auth_fallback_authenticated(self, mock_extract):
        """Fallback: extracts user from request when no require_auth_or_error."""
        mock_user = MagicMock()
        mock_user.is_authenticated = True
        mock_user.user_id = "user-99"
        mock_extract.return_value = mock_user

        mixin = AuthenticatedHandlerMixin()
        mixin.ctx = {}
        result = mixin.require_auth(MagicMock())
        assert result.user_id == "user-99"

    @patch("aragora.server.handlers.mixins.extract_user_from_request")
    def test_require_auth_fallback_unauthenticated(self, mock_extract):
        """Fallback: returns 401 error when user is not authenticated."""
        mock_user = MagicMock()
        mock_user.is_authenticated = False
        mock_extract.return_value = mock_user

        mixin = AuthenticatedHandlerMixin()
        mixin.ctx = {}
        result = mixin.require_auth(MagicMock())
        assert result.status_code == 401
