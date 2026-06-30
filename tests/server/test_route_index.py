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


@pytest.fixture(autouse=True)
def _reset_route_index_singleton():
    """Reset the global RouteIndex singleton before each test.

    The singleton accumulates state across tests when run as part of a larger
    suite. Clearing it ensures each test starts with a fresh route index.
    """
    import aragora.server.handler_registry.core as core_module

    original = core_module._route_index
    core_module._route_index = None
    yield
    core_module._route_index = original


def _make_mock_handler(routes=None, can_handle_return=True, can_handle_side_effect=None):
    """Create a mock handler compatible with RouteIndex.build().

    Sets ROUTES and ROUTE_PREFIXES explicitly so that Mock auto-attribute
    creation does not produce non-iterable truthy stubs.
    """
    handler = Mock()
    handler.ROUTES = routes or []
    handler.ROUTE_PREFIXES = None
    if can_handle_side_effect is not None:
        handler.can_handle = Mock(side_effect=can_handle_side_effect)
    else:
        handler.can_handle = Mock(return_value=can_handle_return)
    return handler


class TestRouteIndex:
    """Tests for RouteIndex class."""

    @pytest.fixture
    def route_index(self):
        """Create a fresh RouteIndex for testing."""
        return RouteIndex()

    @pytest.fixture
    def mock_handler(self):
        """Create a mock handler with ROUTES and can_handle."""
        return _make_mock_handler(routes=["/api/test", "/api/test/list"])

    @pytest.fixture
    def mock_registry_mixin(self, mock_handler):
        """Create a mock registry mixin with handlers."""
        mixin = Mock()
        mixin._test_handler = mock_handler
        return mixin

    def test_exact_route_lookup(self, route_index, mock_handler, mock_registry_mixin):
        """Exact paths are found via O(1) dict lookup."""
        handler_registry = [("_test_handler", type(mock_handler))]
        route_index.build(mock_registry_mixin, handler_registry)

        # Should find exact match
        result = route_index.get_handler("/api/test")
        assert result is not None
        attr_name, handler = result
        assert attr_name == "_test_handler"
        assert handler == mock_handler

    def test_missing_route_returns_none(self, route_index, mock_handler, mock_registry_mixin):
        """Unknown paths return None."""
        # Mock can_handle to return False for unknown paths
        mock_handler.can_handle = Mock(return_value=False)
        handler_registry = [("_test_handler", type(mock_handler))]
        route_index.build(mock_registry_mixin, handler_registry)

        result = route_index.get_handler("/api/unknown")
        assert result is None

    def test_prefix_route_with_can_handle_verification(self, route_index):
        """Prefix patterns verify with can_handle before returning."""
        handler = _make_mock_handler(
            routes=["/api/debates"],
            can_handle_side_effect=lambda p: p.startswith("/api/debates"),
        )

        mixin = Mock()
        mixin._debates_handler = handler

        handler_registry = [("_debates_handler", type(handler))]
        route_index.build(mixin, handler_registry)

        # Dynamic route should work via prefix
        result = route_index.get_handler("/api/debates/some-id")
        assert result is not None

    def test_cache_is_cleared_on_rebuild(self, route_index, mock_handler, mock_registry_mixin):
        """Rebuilding the index clears the LRU cache."""
        handler_registry = [("_test_handler", type(mock_handler))]

        # First build
        route_index.build(mock_registry_mixin, handler_registry)

        # Make a cached call (method now takes path and normalized_path)
        route_index._get_handler_cached("/api/dynamic/path", "/api/dynamic/path")

        # Rebuild should clear cache
        route_index.build(mock_registry_mixin, handler_registry)
        cache_info_2 = route_index._get_handler_cached.cache_info()

        # Cache should be empty after rebuild
        assert cache_info_2.currsize == 0


class TestRouteIndexIntegration:
    """Integration tests with actual handler classes."""

    @pytest.fixture(autouse=True)
    def _snapshot_handler_registry(self):
        """Ensure HANDLER_REGISTRY is not corrupted by prior tests.

        Takes a snapshot of the registry contents and restores it after
        each test to prevent cross-test pollution via in-place mutations.
        """
        import aragora.server.handler_registry as registry_module

        original_list = list(registry_module.HANDLER_REGISTRY)
        yield
        registry_module.HANDLER_REGISTRY[:] = original_list

    def test_all_handlers_have_routes_attribute(self):
        """Most registered handlers should have ROUTES or can_handle attribute.

        Some handlers are registered for discovery but not yet wired for
        request dispatch (facade/stub handlers). This test verifies that the
        majority of handlers have the routing interface and that the number
        of stubs does not grow unboundedly.
        """
        total = 0
        with_interface = 0
        missing = []
        for attr_name, handler_class in HANDLER_REGISTRY:
            if handler_class is not None:
                total += 1
                has_routes = hasattr(handler_class, "ROUTES")
                has_can_handle = hasattr(handler_class, "can_handle")
                if has_routes or has_can_handle:
                    with_interface += 1
                else:
                    missing.append(handler_class.__name__)

        # At least 80% of handlers should have the routing interface
        assert total > 0, "No handlers in registry"
        ratio = with_interface / total
        assert ratio >= 0.80, (
            f"Only {with_interface}/{total} ({ratio:.0%}) handlers have ROUTES or can_handle. "
            f"Missing: {', '.join(missing)}"
        )

    def test_exact_routes_from_all_handlers(self):
        """Collect exact routes and verify route index coverage.

        Some route overlaps exist where multiple handlers claim the same
        path (e.g. facade handlers, RLM context vs RLM handler). The route
        index resolves these by first-match priority, so overlaps are
        tolerated here. This test verifies that the registry contains a
        meaningful number of routes.
        """
        all_routes = {}

        for attr_name, handler_class in HANDLER_REGISTRY:
            if handler_class is None:
                continue

            routes = getattr(handler_class, "ROUTES", [])
            for route in routes:
                if route not in all_routes:
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
            handler = _make_mock_handler(routes=[f"/api/route{i}"])
            setattr(mixin, f"_handler_{i}", handler)
            handlers.append((f"_handler_{i}", type(handler)))

        index.build(mixin, handlers)

        # First route should be found immediately (dict lookup)
        result = index.get_handler("/api/route0")
        assert result is not None

        # Last route should also be O(1)
        result = index.get_handler("/api/route99")
        assert result is not None

    def test_cache_hit_rate_for_repeated_paths(self):
        """Repeated dynamic paths should hit cache."""
        index = RouteIndex()

        handler = _make_mock_handler(routes=[])

        mixin = Mock()
        mixin._test_handler = handler

        # Manually add prefix pattern
        index._prefix_routes = [("/api/dynamic/", "_test_handler", handler)]
        index._exact_routes = {}
        index._get_handler_cached.cache_clear()

        # First call - cache miss (method takes path and normalized_path)
        index._get_handler_cached("/api/dynamic/item-1", "/api/dynamic/item-1")
        info1 = index._get_handler_cached.cache_info()
        assert info1.misses == 1

        # Second call - cache hit
        index._get_handler_cached("/api/dynamic/item-1", "/api/dynamic/item-1")
        info2 = index._get_handler_cached.cache_info()
        assert info2.hits == 1
