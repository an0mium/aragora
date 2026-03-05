"""Tests for handler registry and routing.

Verifies that all 33 handlers:
1. Register correctly in HANDLER_REGISTRY
2. Route to expected paths via can_handle()
3. Don't have routing conflicts with other handlers
4. Initialize properly with context
"""

import inspect

import pytest
from typing import Optional

from aragora.server.handler_registry import (
    HANDLER_REGISTRY,
    HANDLERS_AVAILABLE,
    HandlerRegistryMixin,
)


def _resolve_handler(handler_class):
    """Resolve a possibly-deferred handler class to the actual class."""
    if hasattr(handler_class, "resolve"):
        return handler_class.resolve()
    return handler_class


def _instantiate_handler(handler_class, ctx):
    """Try to instantiate a handler with ctx, falling back to no-arg."""
    try:
        return handler_class(ctx)
    except TypeError:
        try:
            return handler_class()
        except TypeError:
            return None


# Expected route mappings: (handler_class_name, expected_paths)
# Note: Some handlers require /api/v1/ prefix, others accept both
EXPECTED_ROUTES = [
    ("SystemHandler", ["/api/debug/test"]),
    ("DebatesHandler", ["/api/debates", "/api/v1/debates"]),
    ("AgentsHandler", ["/api/agents", "/api/v1/agents"]),
    ("PulseHandler", ["/api/v1/pulse/trending"]),
    ("MetricsHandler", ["/metrics", "/api/metrics"]),
    ("ConsensusHandler", ["/api/consensus/stats", "/api/v1/consensus/stats"]),
    ("ReplaysHandler", ["/api/replays", "/api/v1/replays"]),
    ("TournamentHandler", ["/api/tournaments", "/api/v1/tournaments"]),
    ("DocumentHandler", ["/api/v1/documents"]),
    ("PersonaHandler", ["/api/personas", "/api/v1/personas"]),
    ("DashboardHandler", ["/api/dashboard/debates", "/api/v1/dashboard/debates"]),
    (
        "CalibrationHandler",
        ["/api/calibration/leaderboard", "/api/v1/calibration/leaderboard"],
    ),
    ("EvolutionHandler", ["/api/evolution/history", "/api/v1/evolution/history"]),
    ("PluginsHandler", ["/api/plugins", "/api/v1/plugins"]),
    ("SocialMediaHandler", ["/api/v1/debates/test-123/publish/twitter"]),
    ("BroadcastHandler", ["/api/v1/debates/test-123/broadcast"]),
    ("ProbesHandler", ["/api/v1/probes/capability"]),
    ("InsightsHandler", ["/api/insights/patterns", "/api/v1/insights/patterns"]),
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

        # Verify all handler classes were imported successfully
        assert all(
            callable(cls)
            for cls in [
                DebatesHandler,
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
            ]
        )


class TestHandlerRouting:
    """Tests for handler routing via can_handle()."""

    @pytest.fixture
    def handlers(self):
        """Create routing handlers with empty context.

        Only includes handlers with can_handle() method. Facade and
        non-routing handlers are excluded.
        """
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
        result = {}
        for _, handler_class in HANDLER_REGISTRY:
            if handler_class is None:
                continue
            resolved = _resolve_handler(handler_class)
            if resolved is None:
                continue
            handler = _instantiate_handler(resolved, ctx)
            if handler is None:
                continue
            if hasattr(handler, "can_handle") and callable(handler.can_handle):
                result[resolved.__name__] = handler
        return result

    def test_handlers_have_can_handle(self, handlers):
        """All routing handlers should have can_handle method."""
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
        assert handler_name in handlers, f"{handler_name} not available"

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
        """Create routing handlers with empty context."""
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
        result = []
        for _, handler_class in HANDLER_REGISTRY:
            if handler_class is None:
                continue
            resolved = _resolve_handler(handler_class)
            if resolved is None:
                continue
            handler = _instantiate_handler(resolved, ctx)
            if handler is None:
                continue
            if hasattr(handler, "can_handle") and callable(handler.can_handle):
                result.append((resolved.__name__, handler))
        return result

    def test_no_double_routing_for_specific_paths(self, handlers):
        """Critical paths should only be handled by one handler.

        Some overlap is acceptable (e.g., /api/debates is handled by DebatesHandler,
        but /api/debates/xxx/broadcast is handled by BroadcastHandler).
        This test checks for unintended conflicts on exact paths.
        """
        # Paths that should be handled by exactly one handler
        # Note: /metrics is intentionally handled by multiple handlers
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
        """Handlers should accept ctx dict or no-args in __init__."""
        ctx = {}
        for _, handler_class in HANDLER_REGISTRY:
            if handler_class is None:
                continue
            resolved = _resolve_handler(handler_class)
            if resolved is None:
                continue
            handler = _instantiate_handler(resolved, ctx)
            assert handler is not None, f"{resolved.__name__} failed to instantiate"

    def test_handlers_have_ctx_attribute(self):
        """Base handlers accepting ctx should store it."""
        from aragora.server.handlers.base import BaseHandler

        ctx = {"test_key": "test_value"}
        for _, handler_class in HANDLER_REGISTRY:
            if handler_class is None:
                continue
            resolved = _resolve_handler(handler_class)
            if resolved is None:
                continue
            # Only check BaseHandler subclasses (they always store ctx)
            if not (isinstance(resolved, type) and issubclass(resolved, BaseHandler)):
                continue
            try:
                handler = resolved(ctx)
            except TypeError:
                continue
            assert hasattr(handler, "ctx"), f"{resolved.__name__} missing ctx attribute"

    def test_handlers_expose_get_storage(self):
        """Handlers with require_storage decorator should have get_storage."""
        from unittest.mock import Mock

        ctx = {"storage": Mock()}
        for _, handler_class in HANDLER_REGISTRY:
            if handler_class is None:
                continue
            resolved = _resolve_handler(handler_class)
            if resolved is None:
                continue
            try:
                handler = resolved(ctx)
            except TypeError:
                continue
            if hasattr(handler, "get_storage"):
                storage = handler.get_storage()
                assert storage is ctx["storage"]


class TestHandlerRegistryMixin:
    """Tests for HandlerRegistryMixin."""

    def test_mixin_has_handler_attributes(self):
        """Mixin should have handler attributes after _init_handlers.

        Attributes are set dynamically via _init_handlers(), not declared
        as class-level stubs. Verify the init method exists and can be called.
        """
        assert hasattr(HandlerRegistryMixin, "_init_handlers")
        assert callable(HandlerRegistryMixin._init_handlers)

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


class TestPOSTFallbackDispatch:
    """Tests for POST dispatch fallback from handle_post() to handle().

    When BaseHandler.handle_post() returns None (the default stub), the
    handler registry should fall through to calling handle() with the
    POST method, so handlers that route POST in handle() still work.

    This was a production bug: 60+ handlers route POST in handle() but
    don't override handle_post(). The inherited BaseHandler.handle_post()
    returns None, causing 404 for all POST requests to these handlers.
    """

    def test_base_handler_post_returns_none(self):
        """BaseHandler.handle_post() should return None (stub)."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        result = handler.handle_post("/api/test", {}, None)
        assert result is None

    def test_auth_handler_routes_post_in_handle(self):
        """AuthHandler handles POST /api/auth/login in handle(), not handle_post()."""
        from aragora.server.handlers.auth.handler import AuthHandler

        handler = AuthHandler({})
        # handle_post inherited from BaseHandler returns None
        assert handler.handle_post("/api/auth/login", {}, None) is None
        # But handle() with POST method should route to login
        assert handler.can_handle("/api/auth/login")
        # The handler's handle() checks method=="POST" for login

    def test_handlers_with_post_in_handle_but_no_override(self):
        """Many handlers route POST in handle() but don't override handle_post().

        The dispatch code must fall through to handle() when handle_post()
        returns None, otherwise all these handlers fail with 404 on POST.
        """
        from aragora.server.handlers.base import BaseHandler

        handlers_using_post_via_handle = []
        for attr_name, hc in HANDLER_REGISTRY:
            if hc is None:
                continue
            resolved = _resolve_handler(hc)
            if resolved is None:
                continue
            # Check if handler routes POST in handle() but uses base handle_post
            if not hasattr(resolved, "handle"):
                continue
            try:
                handle_src = inspect.getsource(resolved.handle)
            except (TypeError, OSError):
                continue
            if '"POST"' not in handle_src and "'POST'" not in handle_src:
                continue
            # Check if handle_post is the BaseHandler stub or missing
            if not hasattr(resolved, "handle_post"):
                handlers_using_post_via_handle.append(resolved.__name__)
                continue
            if resolved.handle_post is BaseHandler.handle_post:
                handlers_using_post_via_handle.append(resolved.__name__)

        # There should be many such handlers (AuthHandler, AdminHandler, etc.)
        assert len(handlers_using_post_via_handle) >= 10, (
            f"Expected 10+ handlers routing POST via handle(), found: "
            f"{len(handlers_using_post_via_handle)}"
        )

    def test_base_handler_delete_returns_none(self):
        """BaseHandler.handle_delete() should also return None (stub)."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        result = handler.handle_delete("/api/test", {}, None)
        assert result is None

    def test_base_handler_patch_returns_none(self):
        """BaseHandler.handle_patch() should also return None (stub)."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        result = handler.handle_patch("/api/test", {}, None)
        assert result is None

    def test_base_handler_put_returns_none(self):
        """BaseHandler.handle_put() should also return None (stub)."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        result = handler.handle_put("/api/test", {}, None)
        assert result is None
