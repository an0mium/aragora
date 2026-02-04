"""
Handler mixin classes for common functionality.

This module provides reusable mixins for HTTP handlers:
- PaginatedHandlerMixin: Standardized pagination handling
- CachedHandlerMixin: Response caching with TTL
- AuthenticatedHandlerMixin: Authentication requirement helpers

Usage:
    from aragora.server.handlers.mixins import (
        PaginatedHandlerMixin,
        CachedHandlerMixin,
        AuthenticatedHandlerMixin,
    )

    class MyHandler(BaseHandler, PaginatedHandlerMixin, CachedHandlerMixin):
        def handle(self, path, query_params, handler):
            limit, offset = self.get_pagination(query_params)
            results = self.cached_response(
                cache_key=f"mydata:{limit}:{offset}",
                ttl_seconds=300,
                generator=lambda: expensive_computation(limit, offset),
            )
            return self.paginated_response(results, total=100, limit=limit, offset=offset)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from aragora.server.handlers.admin.cache import get_handler_cache
from aragora.server.handlers.utils.params import get_int_param
from aragora.server.handlers.utils.responses import HandlerResult, error_response, json_response

if TYPE_CHECKING:
    pass


class PaginatedHandlerMixin:
    """Mixin for standardized pagination handling.

    Provides consistent limit/offset extraction with validation and defaults.

    Usage:
        class MyHandler(BaseHandler, PaginatedHandlerMixin):
            def handle(self, path, query_params, handler):
                limit, offset = self.get_pagination(query_params)
                results = self.get_data(limit=limit, offset=offset)
                return self.paginated_response(results, total=100, limit=limit, offset=offset)
    """

    DEFAULT_LIMIT = 20
    MAX_LIMIT = 100
    DEFAULT_OFFSET = 0

    def get_pagination(
        self,
        query_params: dict[str, Any],
        default_limit: int | None = None,
        max_limit: int | None = None,
    ) -> tuple[int, int]:
        """Extract and validate pagination parameters.

        Args:
            query_params: Query parameters dict
            default_limit: Override default limit (default: DEFAULT_LIMIT)
            max_limit: Override max limit (default: MAX_LIMIT)

        Returns:
            Tuple of (limit, offset) with validated bounds
        """
        default_limit = default_limit or self.DEFAULT_LIMIT
        max_limit = max_limit or self.MAX_LIMIT

        limit = get_int_param(query_params, "limit", default_limit)
        offset = get_int_param(query_params, "offset", self.DEFAULT_OFFSET)

        # Clamp values
        limit = max(1, min(limit, max_limit))
        offset = max(0, offset)

        return limit, offset

    def paginated_response(
        self,
        items: list[Any],
        total: int,
        limit: int,
        offset: int,
        items_key: str = "items",
    ) -> HandlerResult:
        """Create a standardized paginated response.

        Args:
            items: List of items for this page
            total: Total count of all items
            limit: Page size used
            offset: Starting offset
            items_key: Key name for items in response (default: "items")

        Returns:
            JSON response with pagination metadata
        """
        return json_response(
            {
                items_key: items,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + len(items) < total,
            }
        )


class CachedHandlerMixin:
    """Mixin for cached response generation.

    Provides a simple interface for caching handler responses with TTL.

    Usage:
        class MyHandler(BaseHandler, CachedHandlerMixin):
            def _get_data(self, key: str):
                return self.cached_response(
                    cache_key=f"mydata:{key}",
                    ttl_seconds=300,
                    generator=lambda: expensive_computation(key),
                )
    """

    def cached_response(
        self,
        cache_key: str,
        ttl_seconds: float,
        generator: Callable[[], Any],
    ) -> Any:
        """Get or generate a cached response.

        Args:
            cache_key: Unique key for this cached item
            ttl_seconds: How long to cache the result
            generator: Callable that generates the value if not cached

        Returns:
            Cached or freshly generated value
        """
        cache = get_handler_cache()
        hit, cached_value = cache.get(cache_key, ttl_seconds)

        if hit:
            return cached_value

        value = generator()
        cache.set(cache_key, value)
        return value

    async def async_cached_response(
        self,
        cache_key: str,
        ttl_seconds: float,
        generator: Callable[[], Any],
    ) -> Any:
        """Async version of cached_response.

        Args:
            cache_key: Unique key for this cached item
            ttl_seconds: How long to cache the result
            generator: Async callable that generates the value if not cached

        Returns:
            Cached or freshly generated value
        """
        cache = get_handler_cache()
        hit, cached_value = cache.get(cache_key, ttl_seconds)

        if hit:
            return cached_value

        value = await generator()
        cache.set(cache_key, value)
        return value


class AuthenticatedHandlerMixin:
    """Mixin for requiring authenticated access.

    Provides standardized authentication extraction and error handling.

    Usage:
        class MyHandler(BaseHandler, AuthenticatedHandlerMixin):
            @require_permission("debates:write")
            def handle_post(self, path, query_params, handler):
                user = self.require_auth(handler)
                if isinstance(user, tuple):  # Error response
                    return user
                # user is now the authenticated context
                return json_response({"user_id": user.user_id})
    """

    def require_auth(self, handler: Any) -> Any:
        """Require authentication and return user context or error.

        Args:
            handler: HTTP request handler with headers

        Returns:
            UserAuthContext if authenticated,
            or HandlerResult with 401 error if not
        """
        # This method is typically overridden or uses BaseHandler's method
        # When used with BaseHandler, call require_auth_or_error instead
        if hasattr(self, "require_auth_or_error"):
            user, err = self.require_auth_or_error(handler)
            if err:
                return err
            return user

        # Fallback implementation
        from aragora.billing.jwt_auth import extract_user_from_request

        user_store = getattr(self, "ctx", {}).get("user_store")
        if hasattr(self.__class__, "user_store"):
            user_store = self.__class__.user_store

        user_ctx = extract_user_from_request(handler, user_store)
        if not user_ctx.is_authenticated:
            return error_response("Authentication required", 401)
        return user_ctx


__all__ = [
    "PaginatedHandlerMixin",
    "CachedHandlerMixin",
    "AuthenticatedHandlerMixin",
]
