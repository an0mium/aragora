"""Tests for handler registry and routing.

Verifies that all 33 handlers:
1. Register correctly in HANDLER_REGISTRY
2. Route to expected paths via can_handle()
3. Don't have routing conflicts with other handlers
4. Initialize properly with context
"""

import pytest
from typing import Optional

from aragora.server.handler_registry import (
    HANDLER_REGISTRY,
    HANDLERS_AVAILABLE,
    HandlerRegistryMixin,
)


# Expected route mappings: (handler_class_name, expected_paths)
# Note: Some handlers only handle specific sub-paths, not the base path
EXPECTED_ROUTES = [
    ("SystemHandler", ["/api/health"]),
    ("DebatesHandler", ["/api/debates", "/api/debates/test-123"]),
    ("AgentsHandler", ["/api/agents", "/api/leaderboard"]),
    ("PulseHandler", ["/api/pulse/trending"]),
    ("MetricsHandler", ["/metrics", "/api/metrics"]),
    ("ConsensusHandler", ["/api/consensus/stats", "/api/consensus/domain/test"]),
    ("ReplaysHandler", ["/api/replays", "/api/replays/test-123"]),
    ("TournamentHandler", ["/api/tournaments"]),
    ("DocumentHandler", ["/api/documents", "/api/documents/test-doc"]),
    ("PersonaHandler", ["/api/personas"]),
    ("DashboardHandler", ["/api/dashboard/debates"]),
    ("CalibrationHandler", ["/api/calibration/leaderboard"]),
    ("EvolutionHandler", ["/api/evolution/history"]),
    ("PluginsHandler", ["/api/plugins", "/api/plugins/test-plugin"]),
    ("SocialMediaHandler", ["/api/debates/test-123/publish/twitter"]),
    ("BroadcastHandler", ["/api/debates/test-123/broadcast"]),
    ("ProbesHandler", ["/api/probes/capability"]),
    ("InsightsHandler", ["/api/insights/patterns"]),
]


class TestHandlerRegistry:
    """Tests for handler registry configuration."""

    def test_handlers_available(self):
        """HANDLERS_AVAILABLE should be True if all imports succeeded."""
        assert HANDLERS_AVAILABLE is True, "Handler imports failed"

    def test_handler_registry_count(self):
        """HANDLER_REGISTRY should contain all 33 handlers."""
        assert len(HANDLER_REGISTRY) >= 33, (
            f"Expected at least 33 handlers, got {len(HANDLER_REGISTRY)}"
        )

    def test_handler_registry_structure(self):
        """Each registry entry should be a (str, class) tuple."""
        for entry in HANDLER_REGISTRY:
            assert isinstance(entry, tuple), f"Entry {entry} is not a tuple"
            assert len(entry) == 2, f"Entry {entry} should have 2 elements"
            attr_name, handler_class = entry
            assert isinstance(attr_name, str), f"Attr name {attr_name} is not a string"
            assert attr_name.startswith("_"), f"Attr name {attr_name} should start with '_'"
            assert attr_name.endswith("_handler"), (
                f"Attr name {attr_name} should end with '_handler'"
            )
            assert handler_class is not None, f"Handler class for {attr_name} is None"

    def test_all_handler_classes_importable(self):
        """All handler classes should be importable."""
        from aragora.server.handlers import (
            SystemHandler,
            DebatesHandler,
            AgentsHandler,
            PulseHandler,
            AnalyticsHandler,
            MetricsHandler,
            ConsensusHandler,
            BeliefHandler,
            CritiqueHandler,
            GenesisHandler,
            ReplaysHandler,
            TournamentHandler,
            MemoryHandler,
            LeaderboardViewHandler,
            DocumentHandler,
            VerificationHandler,
            AuditingHandler,
            RelationshipHandler,
            MomentsHandler,
            PersonaHandler,
            DashboardHandler,
            IntrospectionHandler,
            CalibrationHandler,
            RoutingHandler,
            EvolutionHandler,
            PluginsHandler,
            BroadcastHandler,
            AudioHandler,
            SocialMediaHandler,
            LaboratoryHandler,
            ProbesHandler,
            InsightsHandler,
        )

        # If we get here, all imports succeeded
        assert True


class TestHandlerRouting:
    """Tests for handler routing via can_handle()."""

    @pytest.fixture
    def handlers(self):
        """Create all handlers with empty context."""
        ctx = {
            "storage": None,
            "elo_system": None,
            "nomic_dir": None,
            "debate_embeddings": None,
            "critique_store": None,
            "document_store": None,
            "persona_manager": None,
            "position_ledger": None,
        }
        return {
            handler_class.__name__: handler_class(ctx)
            for _, handler_class in HANDLER_REGISTRY
            if handler_class is not None
        }

    def test_handlers_have_can_handle(self, handlers):
        """All handlers should have can_handle method."""
        for name, handler in handlers.items():
            assert hasattr(handler, "can_handle"), f"{name} missing can_handle method"
            assert callable(handler.can_handle), f"{name}.can_handle is not callable"

    def test_handlers_have_handle(self, handlers):
        """All handlers should have handle method."""
        for name, handler in handlers.items():
            assert hasattr(handler, "handle"), f"{name} missing handle method"
            assert callable(handler.handle), f"{name}.handle is not callable"

    @pytest.mark.parametrize("handler_name,expected_paths", EXPECTED_ROUTES)
    def test_handler_routes_expected_paths(self, handlers, handler_name, expected_paths):
        """Each handler should route its expected paths."""
        if handler_name not in handlers:
            pytest.skip(f"{handler_name} not available")

        handler = handlers[handler_name]
        for path in expected_paths:
            # Try to find at least one path that matches
            if handler.can_handle(path):
                return  # Found a match, test passes

        # If none matched, show which paths were tried
        pytest.fail(f"{handler_name} doesn't handle any of its expected paths: {expected_paths}")


class TestRoutingConflicts:
    """Tests for routing conflicts between handlers."""

    @pytest.fixture
    def handlers(self):
        """Create all handlers with empty context."""
        ctx = {
            "storage": None,
            "elo_system": None,
            "nomic_dir": None,
            "debate_embeddings": None,
            "critique_store": None,
            "document_store": None,
            "persona_manager": None,
            "position_ledger": None,
        }
        return [
            (handler_class.__name__, handler_class(ctx))
            for _, handler_class in HANDLER_REGISTRY
            if handler_class is not None
        ]

    def test_no_double_routing_for_specific_paths(self, handlers):
        """Critical paths should only be handled by one handler.

        Some overlap is acceptable (e.g., /api/debates is handled by DebatesHandler,
        but /api/debates/xxx/broadcast is handled by BroadcastHandler).
        This test checks for unintended conflicts on exact paths.
        """
        # Paths that should be handled by exactly one handler
        unique_paths = [
            "/api/health",
            "/api/agents",
            "/api/pulse",
            "/api/consensus",
            "/api/belief",
            "/api/genesis",
            "/api/replays",
            "/api/tournament",
            "/api/memory",
            "/api/documents",
            "/api/audit",
            "/api/relationships",
            "/api/moments",
            "/api/personas",
            "/api/dashboard",
            "/api/introspection",
            "/api/calibration",
            "/api/routing",
            "/api/evolution",
            "/api/plugins",
            "/api/audio",
            "/api/laboratory",
            "/api/probes",
            "/api/insights",
            "/metrics",
        ]

        conflicts = []
        for path in unique_paths:
            matching_handlers = [name for name, handler in handlers if handler.can_handle(path)]
            if len(matching_handlers) > 1:
                conflicts.append(f"{path}: {matching_handlers}")

        if conflicts:
            pytest.fail("Routing conflicts detected:\n" + "\n".join(conflicts))


class TestHandlerInitialization:
    """Tests for handler initialization."""

    def test_handlers_accept_context_dict(self):
        """All handlers should accept ctx dict in __init__."""
        ctx = {}
        for _, handler_class in HANDLER_REGISTRY:
            if handler_class is None:
                continue
            # Should not raise
            handler = handler_class(ctx)
            assert handler is not None

    def test_handlers_have_ctx_attribute(self):
        """All handlers should store ctx after initialization."""
        ctx = {"test_key": "test_value"}
        for _, handler_class in HANDLER_REGISTRY:
            if handler_class is None:
                continue
            handler = handler_class(ctx)
            assert hasattr(handler, "ctx"), f"{handler_class.__name__} missing ctx attribute"

    def test_handlers_expose_get_storage(self):
        """Handlers with require_storage decorator should have get_storage."""
        from unittest.mock import Mock

        ctx = {"storage": Mock()}
        for _, handler_class in HANDLER_REGISTRY:
            if handler_class is None:
                continue
            handler = handler_class(ctx)
            if hasattr(handler, "get_storage"):
                # Should return the storage from ctx
                storage = handler.get_storage()
                assert storage is ctx["storage"]


class TestHandlerRegistryMixin:
    """Tests for HandlerRegistryMixin."""

    def test_mixin_has_handler_attributes(self):
        """Mixin should have all handler attribute declarations."""
        for attr_name, _ in HANDLER_REGISTRY:
            assert hasattr(HandlerRegistryMixin, attr_name), f"Mixin missing {attr_name} attribute"

    def test_mixin_has_init_handlers(self):
        """Mixin should have _init_handlers classmethod."""
        assert hasattr(HandlerRegistryMixin, "_init_handlers")
        assert callable(HandlerRegistryMixin._init_handlers)

    def test_mixin_has_try_modular_handler(self):
        """Mixin should have _try_modular_handler method."""
        assert hasattr(HandlerRegistryMixin, "_try_modular_handler")

    def test_mixin_has_get_handler_stats(self):
        """Mixin should have _get_handler_stats method."""
        assert hasattr(HandlerRegistryMixin, "_get_handler_stats")


class TestHandlerMethods:
    """Tests for common handler methods."""

    @pytest.fixture
    def sample_handler(self):
        """Create a sample handler for testing."""
        from aragora.server.handlers import DebatesHandler

        return DebatesHandler({})

    def test_handle_returns_handler_result_or_none(self, sample_handler):
        """handle() should return HandlerResult or None."""
        from aragora.server.handlers.base import HandlerResult

        # For a path the handler doesn't handle, should return None
        result = sample_handler.handle("/api/nonexistent", {}, None)
        assert result is None or isinstance(result, HandlerResult)

    def test_handler_result_structure(self, sample_handler):
        """HandlerResult should have required attributes."""
        from aragora.server.handlers.base import HandlerResult

        # Create a minimal result
        result = HandlerResult(
            status_code=200,
            content_type="application/json",
            body=b"{}",
        )
        assert result.status_code == 200
        assert result.content_type == "application/json"
        assert result.body == b"{}"
        assert result.headers is None or isinstance(result.headers, dict)
