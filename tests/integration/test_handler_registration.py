"""Integration tests for Handler Registration system.

Tests the handler registration, routing, and validation system including:
- HANDLER_REGISTRY completeness and correctness
- RouteIndex O(1) lookup functionality
- Handler validation functions
- Handler initialization and lifecycle
- API versioning support
"""

import pytest
from unittest.mock import MagicMock, patch
import json


class TestHandlerRegistry:
    """Test HANDLER_REGISTRY configuration."""

    def test_handler_registry_not_empty(self):
        """HANDLER_REGISTRY should contain handlers."""
        from aragora.server.handler_registry import HANDLER_REGISTRY

        assert len(HANDLER_REGISTRY) > 0
        assert len(HANDLER_REGISTRY) >= 50  # Should have 55+ handlers

    def test_handler_registry_format(self):
        """Each entry should be (attr_name, handler_class) tuple."""
        from aragora.server.handler_registry import HANDLER_REGISTRY

        for entry in HANDLER_REGISTRY:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            attr_name, handler_class = entry
            assert isinstance(attr_name, str)
            assert attr_name.startswith("_")
            assert attr_name.endswith("_handler")

    def test_handler_registry_unique_attr_names(self):
        """Attribute names should be unique."""
        from aragora.server.handler_registry import HANDLER_REGISTRY

        attr_names = [attr for attr, _ in HANDLER_REGISTRY]
        assert len(attr_names) == len(set(attr_names))

    def test_handlers_available_flag(self):
        """HANDLERS_AVAILABLE should be True when handlers import."""
        from aragora.server.handler_registry import HANDLERS_AVAILABLE

        # In a properly configured environment, handlers should be available
        assert HANDLERS_AVAILABLE is True

    def test_handler_classes_are_valid(self):
        """Handler classes should have required methods."""
        from aragora.server.handler_registry import HANDLER_REGISTRY, HANDLERS_AVAILABLE

        if not HANDLERS_AVAILABLE:
            pytest.skip("Handlers not available")

        for attr_name, handler_class in HANDLER_REGISTRY:
            if handler_class is None:
                continue

            # Required methods
            assert hasattr(handler_class, "can_handle"), f"{attr_name} missing can_handle"
            assert hasattr(handler_class, "handle"), f"{attr_name} missing handle"
            assert callable(handler_class.can_handle), f"{attr_name} can_handle not callable"
            assert callable(handler_class.handle), f"{attr_name} handle not callable"


class TestRouteIndex:
    """Test RouteIndex O(1) lookup functionality."""

    def test_route_index_creation(self):
        """RouteIndex should be creatable."""
        from aragora.server.handler_registry import RouteIndex

        index = RouteIndex()
        assert index is not None
        assert hasattr(index, "_exact_routes")
        assert hasattr(index, "_prefix_routes")

    def test_get_route_index_singleton(self):
        """get_route_index should return the same instance."""
        from aragora.server.handler_registry import get_route_index

        index1 = get_route_index()
        index2 = get_route_index()
        assert index1 is index2

    def test_route_index_exact_path_lookup(self):
        """Exact paths should be O(1) lookup."""
        from aragora.server.handler_registry import RouteIndex

        index = RouteIndex()

        # Add a test route directly
        mock_handler = MagicMock()
        index._exact_routes["/api/test"] = ("_test_handler", mock_handler)

        result = index.get_handler("/api/test")
        assert result is not None
        attr_name, handler = result
        assert attr_name == "_test_handler"
        assert handler is mock_handler

    def test_route_index_prefix_matching(self):
        """Prefix patterns should match dynamic routes."""
        from aragora.server.handler_registry import RouteIndex

        index = RouteIndex()

        # Add a prefix route
        mock_handler = MagicMock()
        mock_handler.can_handle = MagicMock(return_value=True)
        index._prefix_routes.append(("/api/prefix/", "_test_handler", mock_handler))

        # Clear LRU cache before test
        index._get_handler_cached.cache_clear()

        result = index.get_handler("/api/prefix/123")
        assert result is not None
        attr_name, handler = result
        assert attr_name == "_test_handler"

    def test_route_index_no_match(self):
        """Non-matching paths should return None."""
        from aragora.server.handler_registry import RouteIndex

        index = RouteIndex()
        index._get_handler_cached.cache_clear()

        result = index.get_handler("/api/nonexistent/path")
        assert result is None


class TestHandlerValidation:
    """Test handler validation functions."""

    def test_validate_handler_class_valid(self):
        """validate_handler_class should pass for valid handlers."""
        from aragora.server.handler_registry import validate_handler_class

        class ValidHandler:
            ROUTES = ["/api/valid"]

            @classmethod
            def can_handle(cls, path):
                return path.startswith("/api/valid")

            def handle(self, path, query, request_handler):
                return None

        errors = validate_handler_class(ValidHandler, "ValidHandler")
        assert len(errors) == 0

    def test_validate_handler_class_missing_methods(self):
        """validate_handler_class should detect missing methods."""
        from aragora.server.handler_registry import validate_handler_class

        class InvalidHandler:
            pass

        errors = validate_handler_class(InvalidHandler, "InvalidHandler")
        assert len(errors) > 0
        assert any("can_handle" in e for e in errors)
        assert any("handle" in e for e in errors)

    def test_validate_handler_class_none(self):
        """validate_handler_class should handle None handler."""
        from aragora.server.handler_registry import validate_handler_class

        errors = validate_handler_class(None, "NoneHandler")
        assert len(errors) > 0
        assert any("None" in e for e in errors)

    def test_validate_handler_instance_valid(self):
        """validate_handler_instance should pass for valid instances."""
        from aragora.server.handler_registry import validate_handler_instance

        class ValidHandler:
            def can_handle(self, path):
                return False

            def handle(self, path, query, request_handler):
                return None

        instance = ValidHandler()
        errors = validate_handler_instance(instance, "ValidHandler")
        assert len(errors) == 0

    def test_validate_handler_instance_broken_can_handle(self):
        """validate_handler_instance should detect broken can_handle."""
        from aragora.server.handler_registry import validate_handler_instance

        class BrokenHandler:
            def can_handle(self, path):
                raise Exception("Broken!")

            def handle(self, path, query, request_handler):
                return None

        instance = BrokenHandler()
        errors = validate_handler_instance(instance, "BrokenHandler")
        assert len(errors) > 0
        assert any("exception" in e.lower() for e in errors)

    def test_validate_all_handlers(self):
        """validate_all_handlers should check all registry entries."""
        from aragora.server.handler_registry import validate_all_handlers, HANDLERS_AVAILABLE

        if not HANDLERS_AVAILABLE:
            pytest.skip("Handlers not available")

        results = validate_all_handlers(raise_on_error=False)

        assert "valid" in results
        assert "invalid" in results
        assert "missing" in results
        assert "status" in results

        # Most handlers should be valid
        assert len(results["valid"]) > 40


class TestHandlerRegistryMixin:
    """Test HandlerRegistryMixin functionality."""

    def test_mixin_has_handler_attributes(self):
        """Mixin should define handler instance attributes."""
        from aragora.server.handler_registry import HandlerRegistryMixin

        # Check that class has handler placeholders
        assert hasattr(HandlerRegistryMixin, "_health_handler")
        assert hasattr(HandlerRegistryMixin, "_debates_handler")
        assert hasattr(HandlerRegistryMixin, "_agents_handler")
        assert hasattr(HandlerRegistryMixin, "_control_plane_handler")

    def test_mixin_init_handlers_method(self):
        """Mixin should have _init_handlers method."""
        from aragora.server.handler_registry import HandlerRegistryMixin

        assert hasattr(HandlerRegistryMixin, "_init_handlers")
        assert callable(HandlerRegistryMixin._init_handlers)

    def test_mixin_try_modular_handler_method(self):
        """Mixin should have _try_modular_handler method."""
        from aragora.server.handler_registry import HandlerRegistryMixin

        assert hasattr(HandlerRegistryMixin, "_try_modular_handler")

    def test_get_handler_stats_uninitialized(self):
        """_get_handler_stats should work when uninitialized."""
        from aragora.server.handler_registry import HandlerRegistryMixin

        class TestMixin(HandlerRegistryMixin):
            pass

        mixin = TestMixin()
        mixin._handlers_initialized = False

        stats = mixin._get_handler_stats()

        assert stats["initialized"] is False
        assert stats["count"] == 0
        assert stats["handlers"] == []


class TestAPIVersioning:
    """Test API versioning support in routing."""

    def test_version_extraction(self):
        """Version should be extracted from paths."""
        from aragora.server.versioning import extract_version
        from aragora.server.versioning.router import APIVersion

        version, is_legacy = extract_version("/api/v1/debates", {})
        assert version == APIVersion.V1
        assert is_legacy is False

    def test_version_strip_prefix(self):
        """Version prefix should be stripped for handler matching."""
        from aragora.server.versioning import strip_version_prefix

        normalized = strip_version_prefix("/api/v1/debates")
        assert normalized == "/api/debates"

        # Non-versioned paths should be unchanged
        normalized = strip_version_prefix("/api/debates")
        assert normalized == "/api/debates"

    def test_version_response_headers(self):
        """Version headers should be generated."""
        from aragora.server.versioning import version_response_headers
        from aragora.server.versioning.router import APIVersion

        headers = version_response_headers(APIVersion.V1, False)
        assert "X-API-Version" in headers
        assert headers["X-API-Version"] == "v1"


class TestHandlerCanHandlePaths:
    """Test that handlers correctly implement can_handle."""

    def test_health_handler_paths(self):
        """HealthHandler should handle health paths."""
        from aragora.server.handler_registry import HealthHandler, HANDLERS_AVAILABLE

        if not HANDLERS_AVAILABLE or HealthHandler is None:
            pytest.skip("HealthHandler not available")

        handler = HealthHandler({})

        assert handler.can_handle("/healthz")
        assert handler.can_handle("/readyz")
        assert handler.can_handle("/api/v1/health")
        assert not handler.can_handle("/api/v1/debates")

    def test_debates_handler_paths(self):
        """DebatesHandler should handle debate paths."""
        from aragora.server.handler_registry import DebatesHandler, HANDLERS_AVAILABLE

        if not HANDLERS_AVAILABLE or DebatesHandler is None:
            pytest.skip("DebatesHandler not available")

        handler = DebatesHandler({})

        assert handler.can_handle("/api/v1/debates")
        assert handler.can_handle("/api/v1/debates/123")
        assert handler.can_handle("/api/v1/search")
        assert not handler.can_handle("/api/v1/agents")

    def test_control_plane_handler_paths(self):
        """ControlPlaneHandler should handle control plane paths."""
        from aragora.server.handler_registry import ControlPlaneHandler, HANDLERS_AVAILABLE

        if not HANDLERS_AVAILABLE or ControlPlaneHandler is None:
            pytest.skip("ControlPlaneHandler not available")

        handler = ControlPlaneHandler({})

        assert handler.can_handle("/api/v1/control-plane/agents")
        assert handler.can_handle("/api/v1/control-plane/tasks")
        assert handler.can_handle("/api/v1/control-plane/health")
        assert handler.can_handle("/api/v1/control-plane/queue")
        assert handler.can_handle("/api/v1/control-plane/metrics")
        assert not handler.can_handle("/api/v1/debates")


class TestHandlerRoutes:
    """Test handler ROUTES attribute."""

    def test_handlers_have_routes(self):
        """Handlers should define ROUTES for exact matching."""
        from aragora.server.handler_registry import HANDLER_REGISTRY, HANDLERS_AVAILABLE

        if not HANDLERS_AVAILABLE:
            pytest.skip("Handlers not available")

        handlers_with_routes = 0

        for attr_name, handler_class in HANDLER_REGISTRY:
            if handler_class is None:
                continue

            if hasattr(handler_class, "ROUTES"):
                routes = handler_class.ROUTES
                # ROUTES can be a list, tuple, or dict (mapping paths to method names)
                assert isinstance(routes, (list, tuple, dict)), (
                    f"{attr_name} ROUTES is {type(routes)}"
                )
                handlers_with_routes += 1

        # Most handlers should have ROUTES defined
        assert handlers_with_routes > 30

    def test_routes_are_valid_paths(self):
        """ROUTES entries should be valid API paths."""
        from aragora.server.handler_registry import HANDLER_REGISTRY, HANDLERS_AVAILABLE

        if not HANDLERS_AVAILABLE:
            pytest.skip("Handlers not available")

        for attr_name, handler_class in HANDLER_REGISTRY:
            if handler_class is None:
                continue

            routes = getattr(handler_class, "ROUTES", [])
            for route in routes:
                assert isinstance(route, str)
                assert route.startswith("/")


class TestHandlerInstantiation:
    """Test handler instantiation with context."""

    def test_handlers_accept_context(self):
        """Handlers should accept context dict."""
        from aragora.server.handler_registry import HANDLER_REGISTRY, HANDLERS_AVAILABLE

        if not HANDLERS_AVAILABLE:
            pytest.skip("Handlers not available")

        ctx = {
            "storage": None,
            "elo_system": None,
            "debate_embeddings": None,
            "document_store": None,
            "nomic_dir": None,
        }

        instantiated = 0

        for attr_name, handler_class in HANDLER_REGISTRY:
            if handler_class is None:
                continue

            try:
                handler = handler_class(ctx)
                assert handler is not None
                instantiated += 1
            except Exception as e:
                # Some handlers may require specific context
                pass

        # Most handlers should instantiate without errors
        assert instantiated > 40

    def test_handler_has_ctx_attribute(self):
        """Instantiated handlers should have ctx attribute."""
        from aragora.server.handler_registry import HealthHandler, HANDLERS_AVAILABLE

        if not HANDLERS_AVAILABLE or HealthHandler is None:
            pytest.skip("HealthHandler not available")

        ctx = {"storage": None}
        handler = HealthHandler(ctx)

        assert hasattr(handler, "ctx")
        assert handler.ctx == ctx


class TestHandlerValidationError:
    """Test HandlerValidationError exception."""

    def test_validation_error_raised(self):
        """validate_all_handlers should raise on error when requested."""
        from aragora.server.handler_registry import (
            HandlerValidationError,
            validate_all_handlers,
        )

        # With a mock that has missing handlers
        with patch(
            "aragora.server.handler_registry.HANDLER_REGISTRY",
            [("_test_handler", None)],
        ):
            with pytest.raises(HandlerValidationError):
                validate_all_handlers(raise_on_error=True)


class TestKnowledgeHandler:
    """Test KnowledgeHandler registration and paths."""

    def test_knowledge_handler_registered(self):
        """KnowledgeHandler should be in registry."""
        from aragora.server.handler_registry import HANDLER_REGISTRY

        handler_names = [attr for attr, _ in HANDLER_REGISTRY]
        assert "_knowledge_handler" in handler_names

    def test_knowledge_handler_paths(self):
        """KnowledgeHandler should handle knowledge paths."""
        from aragora.server.handler_registry import KnowledgeHandler, HANDLERS_AVAILABLE

        if not HANDLERS_AVAILABLE or KnowledgeHandler is None:
            pytest.skip("KnowledgeHandler not available")

        handler = KnowledgeHandler({})

        # KnowledgeHandler handles these specific routes
        assert handler.can_handle("/api/v1/knowledge/facts")
        assert handler.can_handle("/api/v1/knowledge/stats")
        assert handler.can_handle("/api/v1/knowledge/query")
        assert handler.can_handle("/api/v1/knowledge/search")
        assert not handler.can_handle("/api/v1/debates")


class TestWorkflowHandler:
    """Test WorkflowHandler registration and paths."""

    def test_workflow_handler_registered(self):
        """WorkflowHandler should be in registry."""
        from aragora.server.handler_registry import HANDLER_REGISTRY

        handler_names = [attr for attr, _ in HANDLER_REGISTRY]
        assert "_workflow_handler" in handler_names

    def test_workflow_handler_exists(self):
        """WorkflowHandler should be importable."""
        from aragora.server.handler_registry import WorkflowHandler, HANDLERS_AVAILABLE

        if not HANDLERS_AVAILABLE:
            pytest.skip("Handlers not available")

        assert WorkflowHandler is not None


class TestFeaturesHandler:
    """Test FeaturesHandler registration and paths."""

    def test_features_handler_registered(self):
        """FeaturesHandler should be in registry."""
        from aragora.server.handler_registry import HANDLER_REGISTRY

        handler_names = [attr for attr, _ in HANDLER_REGISTRY]
        assert "_features_handler" in handler_names

    def test_features_handler_paths(self):
        """FeaturesHandler should handle features paths."""
        from aragora.server.handler_registry import FeaturesHandler, HANDLERS_AVAILABLE

        if not HANDLERS_AVAILABLE or FeaturesHandler is None:
            pytest.skip("FeaturesHandler not available")

        handler = FeaturesHandler({})

        assert handler.can_handle("/api/v1/features")
        assert not handler.can_handle("/api/v1/debates")
