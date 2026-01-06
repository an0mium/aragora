"""
Tests for PluginsHandler endpoints.

Endpoints tested:
- GET /api/plugins - List all available plugins
- GET /api/plugins/{name} - Get details for a specific plugin
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch

from aragora.server.handlers.plugins import PluginsHandler
from aragora.server.handlers.base import clear_cache


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_plugin_manifest():
    """Create a mock plugin manifest."""
    def create_manifest(name, description="A test plugin"):
        manifest = Mock()
        manifest.name = name
        manifest.description = description
        manifest.version = "1.0.0"
        manifest.author = "Test Author"
        manifest.to_dict.return_value = {
            "name": name,
            "description": description,
            "version": "1.0.0",
            "author": "Test Author",
            "entrypoint": f"{name}/main.py",
            "requirements": ["numpy", "pandas"],
        }
        return manifest
    return create_manifest


@pytest.fixture
def mock_plugin_runner():
    """Create a mock plugin runner."""
    runner = Mock()
    runner._validate_requirements.return_value = (True, [])
    return runner


@pytest.fixture
def mock_registry(mock_plugin_manifest, mock_plugin_runner):
    """Create a mock plugin registry."""
    registry = Mock()

    plugins = [
        mock_plugin_manifest("test-plugin-1", "First test plugin"),
        mock_plugin_manifest("test-plugin-2", "Second test plugin"),
    ]
    registry.list_plugins.return_value = plugins

    def get_plugin(name):
        for p in plugins:
            if p.name == name:
                return p
        return None

    registry.get.side_effect = get_plugin
    registry.get_runner.return_value = mock_plugin_runner
    return registry


@pytest.fixture
def plugins_handler():
    """Create a PluginsHandler with mock dependencies."""
    ctx = {
        "storage": None,
        "elo_system": None,
        "nomic_dir": None,
    }
    return PluginsHandler(ctx)


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    clear_cache()
    yield
    clear_cache()


# ============================================================================
# Route Matching Tests
# ============================================================================

class TestPluginsRouting:
    """Tests for route matching."""

    def test_can_handle_plugins_list(self, plugins_handler):
        assert plugins_handler.can_handle("/api/plugins") is True

    def test_can_handle_plugin_details(self, plugins_handler):
        assert plugins_handler.can_handle("/api/plugins/test-plugin") is True

    def test_can_handle_plugin_with_hyphen(self, plugins_handler):
        assert plugins_handler.can_handle("/api/plugins/my-cool-plugin") is True

    def test_can_handle_plugin_run(self, plugins_handler):
        # POST /api/plugins/{name}/run is now handled by this handler
        assert plugins_handler.can_handle("/api/plugins/test/run") is True

    def test_cannot_handle_unrelated_routes(self, plugins_handler):
        assert plugins_handler.can_handle("/api/plugin") is False
        assert plugins_handler.can_handle("/api/plugins/name/extra/more") is False
        assert plugins_handler.can_handle("/api/agents") is False


# ============================================================================
# GET /api/plugins Tests
# ============================================================================

class TestListPlugins:
    """Tests for GET /api/plugins endpoint."""

    def test_list_plugins_module_unavailable(self, plugins_handler):
        import aragora.server.handlers.plugins as mod
        original = mod.PLUGINS_AVAILABLE
        mod.PLUGINS_AVAILABLE = False
        try:
            result = plugins_handler.handle("/api/plugins", {}, None)
            assert result is not None
            assert result.status_code == 503
            data = json.loads(result.body)
            assert "not available" in data["error"].lower()
        finally:
            mod.PLUGINS_AVAILABLE = original

    def test_list_plugins_success(self, plugins_handler, mock_registry):
        import aragora.server.handlers.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        with patch.object(mod, 'get_registry', return_value=mock_registry):
            result = plugins_handler.handle("/api/plugins", {}, None)

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert "plugins" in data
            assert data["count"] == 2
            assert len(data["plugins"]) == 2

    def test_list_plugins_empty(self, plugins_handler, mock_registry):
        import aragora.server.handlers.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        mock_registry.list_plugins.return_value = []

        with patch.object(mod, 'get_registry', return_value=mock_registry):
            result = plugins_handler.handle("/api/plugins", {}, None)

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["plugins"] == []
            assert data["count"] == 0


# ============================================================================
# GET /api/plugins/{name} Tests
# ============================================================================

class TestGetPluginDetails:
    """Tests for GET /api/plugins/{name} endpoint."""

    def test_get_plugin_module_unavailable(self, plugins_handler):
        import aragora.server.handlers.plugins as mod
        original = mod.PLUGINS_AVAILABLE
        mod.PLUGINS_AVAILABLE = False
        try:
            result = plugins_handler.handle("/api/plugins/test-plugin", {}, None)
            assert result is not None
            assert result.status_code == 503
        finally:
            mod.PLUGINS_AVAILABLE = original

    def test_get_plugin_success(self, plugins_handler, mock_registry):
        import aragora.server.handlers.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        with patch.object(mod, 'get_registry', return_value=mock_registry):
            result = plugins_handler.handle("/api/plugins/test-plugin-1", {}, None)

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["name"] == "test-plugin-1"
            assert data["requirements_satisfied"] is True
            assert data["missing_requirements"] == []

    def test_get_plugin_not_found(self, plugins_handler, mock_registry):
        import aragora.server.handlers.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        with patch.object(mod, 'get_registry', return_value=mock_registry):
            result = plugins_handler.handle("/api/plugins/nonexistent", {}, None)

            assert result is not None
            assert result.status_code == 404
            data = json.loads(result.body)
            assert "not found" in data["error"].lower()

    def test_get_plugin_invalid_name(self, plugins_handler):
        result = plugins_handler.handle("/api/plugins/test..admin", {}, None)
        assert result is not None
        assert result.status_code == 400

    def test_get_plugin_with_missing_requirements(self, plugins_handler, mock_registry, mock_plugin_runner):
        import aragora.server.handlers.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        mock_plugin_runner._validate_requirements.return_value = (False, ["missing-lib"])

        with patch.object(mod, 'get_registry', return_value=mock_registry):
            result = plugins_handler.handle("/api/plugins/test-plugin-1", {}, None)

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["requirements_satisfied"] is False
            assert "missing-lib" in data["missing_requirements"]


# ============================================================================
# Security Tests
# ============================================================================

class TestPluginsSecurity:
    """Security tests for plugins endpoints."""

    def test_path_traversal_blocked(self, plugins_handler):
        result = plugins_handler.handle("/api/plugins/..%2F..%2Fetc", {}, None)
        assert result.status_code == 400

    def test_sql_injection_blocked(self, plugins_handler):
        result = plugins_handler.handle("/api/plugins/'; DROP TABLE plugins;--", {}, None)
        assert result.status_code == 400

    def test_xss_blocked(self, plugins_handler):
        result = plugins_handler.handle("/api/plugins/<script>alert(1)</script>", {}, None)
        # Path may not match at all (None) or should be rejected (400)
        assert result is None or result.status_code == 400

    def test_valid_plugin_names_accepted(self, plugins_handler, mock_registry):
        import aragora.server.handlers.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        valid_names = ["test-plugin", "plugin_name", "plugin123"]
        for name in valid_names:
            with patch.object(mod, 'get_registry', return_value=mock_registry):
                result = plugins_handler.handle(f"/api/plugins/{name}", {}, None)
                # Should not be 400 (validation error)
                assert result.status_code != 400


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestPluginsErrorHandling:
    """Tests for error handling."""

    def test_handle_returns_none_for_unhandled_route(self, plugins_handler):
        result = plugins_handler.handle("/api/other/endpoint", {}, None)
        assert result is None

    def test_list_plugins_exception(self, plugins_handler):
        import aragora.server.handlers.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        mock_registry = Mock()
        mock_registry.list_plugins.side_effect = Exception("Database error")

        with patch.object(mod, 'get_registry', return_value=mock_registry):
            result = plugins_handler.handle("/api/plugins", {}, None)

            assert result is not None
            assert result.status_code == 500

    def test_get_plugin_exception(self, plugins_handler):
        import aragora.server.handlers.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        mock_registry = Mock()
        mock_registry.get.side_effect = Exception("Database error")

        with patch.object(mod, 'get_registry', return_value=mock_registry):
            result = plugins_handler.handle("/api/plugins/test", {}, None)

            assert result is not None
            assert result.status_code == 500


# ============================================================================
# Edge Cases
# ============================================================================

class TestPluginsEdgeCases:
    """Tests for edge cases."""

    def test_empty_plugin_name(self, plugins_handler):
        # Empty plugin name shouldn't match
        result = plugins_handler.handle("/api/plugins/", {}, None)
        # Should not be handled (None or 400)
        assert result is None or result.status_code == 400

    def test_very_long_plugin_name(self, plugins_handler):
        long_name = "a" * 1000
        result = plugins_handler.handle(f"/api/plugins/{long_name}", {}, None)
        # Should handle gracefully
        assert result is not None

    def test_plugin_without_runner(self, plugins_handler, mock_registry, mock_plugin_manifest):
        import aragora.server.handlers.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        # No runner available
        mock_registry.get_runner.return_value = None

        with patch.object(mod, 'get_registry', return_value=mock_registry):
            result = plugins_handler.handle("/api/plugins/test-plugin-1", {}, None)

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            # Should return manifest without requirements info
            assert data["name"] == "test-plugin-1"
            assert "requirements_satisfied" not in data
