"""Integration tests for server startup and handler initialization.

These tests verify that UnifiedHandler._init_handlers() completes without
error, that the route index builds successfully, and that handler coverage
check passes. External deps (databases, APIs) are mocked but real handler
initialization is tested.

This would have caught the _DeferredImport bug before it hit CI.

Marked with @pytest.mark.integration (auto-applied by conftest).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handler_registry import (
    HANDLER_REGISTRY,
    HANDLERS_AVAILABLE,
    HandlerRegistryMixin,
)
from aragora.server.handler_registry.core import (
    RouteIndex,
    _DeferredImport,
    filter_registry_by_tier,
    get_active_tiers,
    validate_handlers_on_init,
)


class TestInitHandlersLifecycle:
    """Test that _init_handlers completes without error."""

    def _make_test_mixin_class(self):
        """Create a fresh HandlerRegistryMixin subclass for isolated testing.

        Each test gets its own class to avoid cross-test state leakage
        from class-level attributes set by _init_handlers.
        """
        class TestMixin(HandlerRegistryMixin):
            _handlers_initialized = False
            _init_lock = __import__("threading").Lock()

            # Provide stubs for attributes _init_handlers reads
            storage = None
            stream_emitter = None
            control_plane_stream = None
            nomic_loop_stream = None
            elo_system = None
            nomic_state_file = None
            debate_embeddings = None
            critique_store = None
            document_store = None
            persona_manager = None
            position_ledger = None
            user_store = None
            continuum_memory = None
            cross_debate_memory = None
            knowledge_mound = None

        return TestMixin

    def test_init_handlers_completes(self):
        """_init_handlers must complete without raising."""
        TestMixin = self._make_test_mixin_class()
        TestMixin._init_handlers()
        assert TestMixin._handlers_initialized is True

    def test_init_handlers_is_idempotent(self):
        """Calling _init_handlers twice should not error or re-initialize."""
        TestMixin = self._make_test_mixin_class()
        TestMixin._init_handlers()
        assert TestMixin._handlers_initialized is True

        # Second call should be a no-op (fast path)
        TestMixin._init_handlers()
        assert TestMixin._handlers_initialized is True

    def test_init_handlers_sets_handler_instances(self):
        """After init, handler instances should be set as class attributes."""
        TestMixin = self._make_test_mixin_class()
        TestMixin._init_handlers()

        # Check that at least some core handlers are initialized
        initialized_count = 0
        for attr_name, _ in HANDLER_REGISTRY:
            handler = getattr(TestMixin, attr_name, None)
            if handler is not None:
                initialized_count += 1

        assert initialized_count > 50, (
            f"Only {initialized_count} handlers initialized, expected 50+"
        )

    def test_init_handlers_creates_valid_instances(self):
        """Initialized handler instances should pass validation."""
        TestMixin = self._make_test_mixin_class()
        TestMixin._init_handlers()

        active_registry = filter_registry_by_tier(HANDLER_REGISTRY)
        results = validate_handlers_on_init(TestMixin, active_registry)

        assert len(results["valid"]) > 50, (
            f"Only {results['valid'][:5]}... ({len(results['valid'])}) handlers valid"
        )

    def test_handlers_available_flag_true(self):
        """HANDLERS_AVAILABLE should be True in normal environment."""
        assert HANDLERS_AVAILABLE is True


class TestRouteIndexBuild:
    """Test that route index builds successfully after handler init."""

    def _init_mixin_and_get_index(self):
        """Helper: initialize handlers and return a fresh RouteIndex."""
        class TestMixin(HandlerRegistryMixin):
            _handlers_initialized = False
            _init_lock = __import__("threading").Lock()
            storage = None
            stream_emitter = None
            control_plane_stream = None
            nomic_loop_stream = None
            elo_system = None
            nomic_state_file = None
            debate_embeddings = None
            critique_store = None
            document_store = None
            persona_manager = None
            position_ledger = None
            user_store = None
            continuum_memory = None
            cross_debate_memory = None
            knowledge_mound = None

        TestMixin._init_handlers()

        index = RouteIndex()
        active_registry = filter_registry_by_tier(HANDLER_REGISTRY)
        index.build(TestMixin, active_registry)
        return index

    def test_route_index_builds_without_error(self):
        """Building a RouteIndex from initialized handlers must not crash."""
        index = self._init_mixin_and_get_index()
        assert index is not None

    def test_route_index_has_exact_routes(self):
        """After build, route index should have exact routes populated."""
        index = self._init_mixin_and_get_index()
        assert len(index._exact_routes) > 0, "No exact routes registered"

    def test_route_index_has_prefix_routes(self):
        """After build, route index should have prefix routes populated."""
        index = self._init_mixin_and_get_index()
        assert len(index._prefix_routes) > 0, "No prefix routes registered"

    def test_route_index_resolves_health(self):
        """Route index must resolve /healthz to health handler."""
        index = self._init_mixin_and_get_index()
        result = index.get_handler("/healthz")
        assert result is not None, "/healthz did not resolve"
        attr_name, handler = result
        assert "health" in attr_name.lower(), (
            f"/healthz resolved to unexpected handler: {attr_name}"
        )

    def test_route_index_resolves_debates(self):
        """Route index must resolve /api/debates to debates handler."""
        index = self._init_mixin_and_get_index()
        result = index.get_handler("/api/debates")
        assert result is not None, "/api/debates did not resolve"

    def test_route_index_resolves_versioned_path(self):
        """Route index must resolve versioned paths like /api/v1/debates."""
        index = self._init_mixin_and_get_index()
        result = index.get_handler("/api/v1/debates")
        assert result is not None, "/api/v1/debates did not resolve"

    def test_route_index_returns_none_for_unknown(self):
        """Route index must return None for unregistered paths."""
        index = self._init_mixin_and_get_index()
        result = index.get_handler("/api/this-path-does-not-exist-xyz123")
        assert result is None


class TestHandlerCoverageOnInit:
    """Test that check_handler_coverage succeeds during init flow."""

    def test_coverage_check_matches_init_flow(self):
        """Simulate the check_handler_coverage call that happens inside _init_handlers."""
        from aragora.server.handler_registry.core import check_handler_coverage

        active_registry = filter_registry_by_tier(HANDLER_REGISTRY)
        # This is exactly what _init_handlers does -- must not raise
        check_handler_coverage(active_registry)

    def test_coverage_check_with_all_tiers(self):
        """Handler coverage check should pass with all tiers active."""
        from aragora.server.handler_registry.core import check_handler_coverage

        all_tiers = {"core", "extended", "optional", "enterprise", "experimental"}
        full_registry = filter_registry_by_tier(HANDLER_REGISTRY, all_tiers)
        check_handler_coverage(full_registry)


class TestGetHandlerStats:
    """Test _get_handler_stats after initialization."""

    def test_stats_after_init(self):
        """_get_handler_stats should return correct counts after init."""
        class TestMixin(HandlerRegistryMixin):
            _handlers_initialized = False
            _init_lock = __import__("threading").Lock()
            storage = None
            stream_emitter = None
            control_plane_stream = None
            nomic_loop_stream = None
            elo_system = None
            nomic_state_file = None
            debate_embeddings = None
            critique_store = None
            document_store = None
            persona_manager = None
            position_ledger = None
            user_store = None
            continuum_memory = None
            cross_debate_memory = None
            knowledge_mound = None

        TestMixin._init_handlers()
        instance = TestMixin()
        stats = instance._get_handler_stats()

        assert stats["initialized"] is True
        assert stats["count"] > 50
        assert len(stats["handlers"]) > 50
        assert all(isinstance(name, str) for name in stats["handlers"])

    def test_stats_before_init(self):
        """_get_handler_stats should show uninitialized state."""
        class FreshMixin(HandlerRegistryMixin):
            _handlers_initialized = False

        instance = FreshMixin()
        stats = instance._get_handler_stats()

        assert stats["initialized"] is False
        assert stats["count"] == 0
        assert stats["handlers"] == []


class TestTierFilteredInit:
    """Test handler initialization with different tier configurations."""

    def test_core_only_tier_inits_fewer_handlers(self):
        """Core-only tier should initialize fewer handlers than all tiers."""
        core_registry = filter_registry_by_tier(HANDLER_REGISTRY, {"core"})
        all_registry = filter_registry_by_tier(
            HANDLER_REGISTRY,
            {"core", "extended", "optional", "enterprise", "experimental"},
        )

        assert len(core_registry) < len(all_registry), (
            f"Core ({len(core_registry)}) should be smaller than "
            f"all ({len(all_registry)})"
        )
        assert len(core_registry) >= 5, "Core tier too small"

    def test_core_always_includes_health(self):
        """Core tier must always include the health handler."""
        core_registry = filter_registry_by_tier(HANDLER_REGISTRY, {"core"})
        attr_names = [name for name, _ in core_registry]
        assert "_health_handler" in attr_names

    def test_core_always_includes_debates(self):
        """Core tier must always include the debates handler."""
        core_registry = filter_registry_by_tier(HANDLER_REGISTRY, {"core"})
        attr_names = [name for name, _ in core_registry]
        assert "_debates_handler" in attr_names
