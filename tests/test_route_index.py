"""
Tests for RouteIndex O(1) handler dispatch.

Verifies:
- Exact path matching
- Prefix pattern matching
- LRU cache behavior
- Fallback to handler iteration
"""

import pytest
from unittest.mock import Mock, MagicMock

from aragora.server.handler_registry import (
    RouteIndex,
    get_route_index,
    HANDLER_REGISTRY,
)


class TestRouteIndex:
    """Tests for RouteIndex class."""

    @pytest.fixture
    def route_index(self):
        """Create a fresh RouteIndex for testing."""
        return RouteIndex()

    @pytest.fixture
    def mock_handler(self):
        """Create a mock handler with ROUTES and can_handle."""
        handler = Mock()
        handler.ROUTES = ["/api/test", "/api/test/list"]
        handler.can_handle = Mock(return_value=True)
        return handler

    @pytest.fixture
    def mock_registry_mixin(self, mock_handler):
        """Create a mock registry mixin with handlers."""
        mixin = Mock()
        mixin._test_handler = mock_handler
        return mixin

    def test_exact_route_lookup(self, route_index, mock_handler, mock_registry_mixin):
        """Exact paths are found via O(1) dict lookup."""
        # Patch HANDLER_REGISTRY for this test
        import aragora.server.handler_registry as registry_module
        original_registry = registry_module.HANDLER_REGISTRY

        try:
            registry_module.HANDLER_REGISTRY = [("_test_handler", type(mock_handler))]
            route_index.build(mock_registry_mixin)

            # Should find exact match
            result = route_index.get_handler("/api/test")
            assert result is not None
            attr_name, handler = result
            assert attr_name == "_test_handler"
            assert handler == mock_handler
        finally:
            registry_module.HANDLER_REGISTRY = original_registry

    def test_missing_route_returns_none(self, route_index, mock_handler, mock_registry_mixin):
        """Unknown paths return None."""
        import aragora.server.handler_registry as registry_module
        original_registry = registry_module.HANDLER_REGISTRY

        try:
            registry_module.HANDLER_REGISTRY = [("_test_handler", type(mock_handler))]
            # Mock can_handle to return False for unknown paths
            mock_handler.can_handle = Mock(return_value=False)
            route_index.build(mock_registry_mixin)

            result = route_index.get_handler("/api/unknown")
            assert result is None
        finally:
            registry_module.HANDLER_REGISTRY = original_registry

    def test_prefix_route_with_can_handle_verification(self, route_index):
        """Prefix patterns verify with can_handle before returning."""
        handler = Mock()
        handler.ROUTES = ["/api/debates"]
        handler.can_handle = Mock(side_effect=lambda p: p.startswith("/api/debates"))

        mixin = Mock()
        mixin._debates_handler = handler

        import aragora.server.handler_registry as registry_module
        original_registry = registry_module.HANDLER_REGISTRY

        try:
            registry_module.HANDLER_REGISTRY = [("_debates_handler", type(handler))]
            route_index.build(mixin)

            # Dynamic route should work via prefix
            result = route_index.get_handler("/api/debates/some-id")
            assert result is not None
        finally:
            registry_module.HANDLER_REGISTRY = original_registry

    def test_cache_is_cleared_on_rebuild(self, route_index, mock_handler, mock_registry_mixin):
        """Rebuilding the index clears the LRU cache."""
        import aragora.server.handler_registry as registry_module
        original_registry = registry_module.HANDLER_REGISTRY

        try:
            registry_module.HANDLER_REGISTRY = [("_test_handler", type(mock_handler))]

            # First build
            route_index.build(mock_registry_mixin)
            cache_info_1 = route_index._get_handler_cached.cache_info()

            # Make a cached call
            route_index._get_handler_cached("/api/dynamic/path")

            # Rebuild should clear cache
            route_index.build(mock_registry_mixin)
            cache_info_2 = route_index._get_handler_cached.cache_info()

            # Cache should be empty after rebuild
            assert cache_info_2.currsize == 0
        finally:
            registry_module.HANDLER_REGISTRY = original_registry


class TestRouteIndexIntegration:
    """Integration tests with actual handler classes."""

    def test_all_handlers_have_routes_attribute(self):
        """All registered handlers should have ROUTES attribute."""
        for attr_name, handler_class in HANDLER_REGISTRY:
            if handler_class is not None:
                # Handler classes should define ROUTES
                assert hasattr(handler_class, 'ROUTES') or hasattr(handler_class, 'can_handle'), \
                    f"{handler_class.__name__} missing ROUTES or can_handle"

    def test_exact_routes_from_all_handlers(self):
        """Collect and verify all exact routes are unique."""
        all_routes = {}

        for attr_name, handler_class in HANDLER_REGISTRY:
            if handler_class is None:
                continue

            routes = getattr(handler_class, 'ROUTES', [])
            for route in routes:
                if route in all_routes:
                    # Same handler can have overlapping routes
                    if all_routes[route] != attr_name:
                        pytest.fail(
                            f"Route conflict: {route} claimed by both "
                            f"{all_routes[route]} and {attr_name}"
                        )
                all_routes[route] = attr_name

        # Verify we collected some routes
        assert len(all_routes) > 0, "No routes found in handlers"


class TestRouteIndexPerformance:
    """Performance characteristics of RouteIndex."""

    def test_exact_match_is_o1(self):
        """Exact matches should be O(1) regardless of handler count."""
        index = RouteIndex()

        # Create many mock handlers
        mixin = Mock()
        handlers = []

        for i in range(100):
            handler = Mock()
            handler.ROUTES = [f"/api/route{i}"]
            handler.can_handle = Mock(return_value=True)
            setattr(mixin, f"_handler_{i}", handler)
            handlers.append((f"_handler_{i}", type(handler)))

        import aragora.server.handler_registry as registry_module
        original_registry = registry_module.HANDLER_REGISTRY

        try:
            registry_module.HANDLER_REGISTRY = handlers
            index.build(mixin)

            # First route should be found immediately (dict lookup)
            result = index.get_handler("/api/route0")
            assert result is not None

            # Last route should also be O(1)
            result = index.get_handler("/api/route99")
            assert result is not None
        finally:
            registry_module.HANDLER_REGISTRY = original_registry

    def test_cache_hit_rate_for_repeated_paths(self):
        """Repeated dynamic paths should hit cache."""
        index = RouteIndex()

        handler = Mock()
        handler.ROUTES = []
        handler.can_handle = Mock(return_value=True)

        mixin = Mock()
        mixin._test_handler = handler

        import aragora.server.handler_registry as registry_module
        original_registry = registry_module.HANDLER_REGISTRY
        original_prefix = getattr(registry_module.RouteIndex, '_prefix_routes', None)

        try:
            registry_module.HANDLER_REGISTRY = [("_test_handler", type(handler))]

            # Manually add prefix pattern
            index._prefix_routes = [("/api/dynamic/", "_test_handler", handler)]
            index._exact_routes = {}
            index._get_handler_cached.cache_clear()

            # First call - cache miss
            index._get_handler_cached("/api/dynamic/item-1")
            info1 = index._get_handler_cached.cache_info()
            assert info1.misses == 1

            # Second call - cache hit
            index._get_handler_cached("/api/dynamic/item-1")
            info2 = index._get_handler_cached.cache_info()
            assert info2.hits == 1

        finally:
            registry_module.HANDLER_REGISTRY = original_registry
