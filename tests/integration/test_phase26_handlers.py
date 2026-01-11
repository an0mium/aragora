"""Integration tests for Phase 26: Handler Consolidation.

Tests the handler mixins functionality.
"""

import pytest
from unittest.mock import MagicMock


class TestPaginatedHandlerMixin:
    """Test PaginatedHandlerMixin functionality."""

    def test_get_pagination_defaults(self):
        """get_pagination() should return default values."""
        from aragora.server.handlers.base import PaginatedHandlerMixin

        mixin = PaginatedHandlerMixin()
        limit, offset = mixin.get_pagination({})

        assert limit == mixin.DEFAULT_LIMIT
        assert offset == mixin.DEFAULT_OFFSET

    def test_get_pagination_with_params(self):
        """get_pagination() should extract params from query."""
        from aragora.server.handlers.base import PaginatedHandlerMixin

        mixin = PaginatedHandlerMixin()
        limit, offset = mixin.get_pagination({"limit": ["50"], "offset": ["10"]})

        assert limit == 50
        assert offset == 10

    def test_get_pagination_clamps_values(self):
        """get_pagination() should clamp values to bounds."""
        from aragora.server.handlers.base import PaginatedHandlerMixin

        mixin = PaginatedHandlerMixin()

        # Test max clamping
        limit, offset = mixin.get_pagination({"limit": ["500"]})
        assert limit == mixin.MAX_LIMIT

        # Test min clamping
        limit, offset = mixin.get_pagination({"limit": ["0"]})
        assert limit == 1  # Min is 1

    def test_paginated_response_structure(self):
        """paginated_response() should return correct structure."""
        from aragora.server.handlers.base import PaginatedHandlerMixin, HandlerResult

        mixin = PaginatedHandlerMixin()
        items = [{"id": 1}, {"id": 2}]

        result = mixin.paginated_response(
            items=items,
            total=10,
            limit=2,
            offset=0,
        )

        # Result should be a HandlerResult
        assert isinstance(result, HandlerResult)
        assert result.status_code == 200

        import json
        data = json.loads(result.body)

        assert data["items"] == items
        assert data["total"] == 10
        assert data["limit"] == 2
        assert data["offset"] == 0
        assert data["has_more"] is True


class TestCachedHandlerMixin:
    """Test CachedHandlerMixin functionality."""

    def test_cached_response_caches_value(self):
        """cached_response() should cache and return values."""
        from aragora.server.handlers.base import CachedHandlerMixin

        mixin = CachedHandlerMixin()
        call_count = 0

        def generator():
            nonlocal call_count
            call_count += 1
            return {"data": "test"}

        # First call should invoke generator
        result1 = mixin.cached_response("test_key", 60.0, generator)
        assert result1 == {"data": "test"}
        assert call_count == 1

        # Second call should return cached value
        result2 = mixin.cached_response("test_key", 60.0, generator)
        assert result2 == {"data": "test"}
        assert call_count == 1  # Generator not called again


class TestAuthenticatedHandlerMixin:
    """Test AuthenticatedHandlerMixin functionality."""

    def test_require_auth_without_basehandler(self):
        """require_auth() should work standalone."""
        from aragora.server.handlers.base import AuthenticatedHandlerMixin, HandlerResult

        mixin = AuthenticatedHandlerMixin()
        mixin.ctx = {}

        # Mock handler without auth
        handler = MagicMock()
        handler.headers = {}

        result = mixin.require_auth(handler)

        # Should return error HandlerResult for unauthenticated
        assert isinstance(result, HandlerResult)
        assert result.status_code == 401


class TestMixinInheritance:
    """Test that mixins work when combined with BaseHandler."""

    def test_combined_inheritance(self):
        """Mixins should work when combined with BaseHandler."""
        from aragora.server.handlers.base import (
            BaseHandler,
            PaginatedHandlerMixin,
            CachedHandlerMixin,
        )

        class TestHandler(BaseHandler, PaginatedHandlerMixin, CachedHandlerMixin):
            pass

        # Should be able to create instance
        handler = TestHandler({"storage": None})
        assert handler is not None

        # Should have mixin methods
        assert hasattr(handler, "get_pagination")
        assert hasattr(handler, "paginated_response")
        assert hasattr(handler, "cached_response")

        # Should also have BaseHandler methods
        assert hasattr(handler, "get_storage")
        assert hasattr(handler, "extract_path_param")
