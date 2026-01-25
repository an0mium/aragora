"""
Tests for PluginsHandler endpoints.

Endpoints tested:
- GET /api/v1/plugins - List all available plugins
- GET /api/v1/plugins/{name} - Get details for a specific plugin
"""

import json
import pytest
from unittest.mock import Mock, patch

from aragora.server.handlers.features import PluginsHandler
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
        assert plugins_handler.can_handle("/api/v1/plugins") is True

    def test_can_handle_plugin_details(self, plugins_handler):
        assert plugins_handler.can_handle("/api/v1/plugins/test-plugin") is True

    def test_can_handle_plugin_with_hyphen(self, plugins_handler):
        assert plugins_handler.can_handle("/api/v1/plugins/my-cool-plugin") is True

    def test_can_handle_plugin_run(self, plugins_handler):
        # POST /api/v1/plugins/{name}/run is now handled by this handler
        assert plugins_handler.can_handle("/api/v1/plugins/test/run") is True

    def test_cannot_handle_unrelated_routes(self, plugins_handler):
        assert plugins_handler.can_handle("/api/v1/plugin") is False
        assert plugins_handler.can_handle("/api/v1/plugins/name/extra/more") is False
        assert plugins_handler.can_handle("/api/v1/agents") is False


# ============================================================================
# GET /api/v1/plugins Tests
# ============================================================================


class TestListPlugins:
    """Tests for GET /api/v1/plugins endpoint."""

    def test_list_plugins_module_unavailable(self, plugins_handler):
        import aragora.server.handlers.features.plugins as mod

        original = mod.PLUGINS_AVAILABLE
        mod.PLUGINS_AVAILABLE = False
        try:
            result = plugins_handler.handle("/api/v1/plugins", {}, None)
            assert result is not None
            assert result.status_code == 503
            data = json.loads(result.body)
            assert "not available" in data["error"].lower()
        finally:
            mod.PLUGINS_AVAILABLE = original

    def test_list_plugins_success(self, plugins_handler, mock_registry):
        import aragora.server.handlers.features.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        with patch.object(mod, "get_registry", return_value=mock_registry):
            result = plugins_handler.handle("/api/v1/plugins", {}, None)

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert "plugins" in data
            assert data["count"] == 2
            assert len(data["plugins"]) == 2

    def test_list_plugins_empty(self, plugins_handler, mock_registry):
        import aragora.server.handlers.features.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        mock_registry.list_plugins.return_value = []

        with patch.object(mod, "get_registry", return_value=mock_registry):
            result = plugins_handler.handle("/api/v1/plugins", {}, None)

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["plugins"] == []
            assert data["count"] == 0


# ============================================================================
# GET /api/v1/plugins/{name} Tests
# ============================================================================


class TestGetPluginDetails:
    """Tests for GET /api/v1/plugins/{name} endpoint."""

    def test_get_plugin_module_unavailable(self, plugins_handler):
        import aragora.server.handlers.features.plugins as mod

        original = mod.PLUGINS_AVAILABLE
        mod.PLUGINS_AVAILABLE = False
        try:
            result = plugins_handler.handle("/api/v1/plugins/test-plugin", {}, None)
            assert result is not None
            assert result.status_code == 503
        finally:
            mod.PLUGINS_AVAILABLE = original

    def test_get_plugin_success(self, plugins_handler, mock_registry):
        import aragora.server.handlers.features.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        with patch.object(mod, "get_registry", return_value=mock_registry):
            result = plugins_handler.handle("/api/v1/plugins/test-plugin-1", {}, None)

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["name"] == "test-plugin-1"
            assert data["requirements_satisfied"] is True
            assert data["missing_requirements"] == []

    def test_get_plugin_not_found(self, plugins_handler, mock_registry):
        import aragora.server.handlers.features.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        with patch.object(mod, "get_registry", return_value=mock_registry):
            result = plugins_handler.handle("/api/v1/plugins/nonexistent", {}, None)

            assert result is not None
            assert result.status_code == 404
            data = json.loads(result.body)
            assert "not found" in data["error"].lower()

    def test_get_plugin_invalid_name(self, plugins_handler):
        result = plugins_handler.handle("/api/v1/plugins/test..admin", {}, None)
        assert result is not None
        assert result.status_code == 400

    def test_get_plugin_with_missing_requirements(
        self, plugins_handler, mock_registry, mock_plugin_runner
    ):
        import aragora.server.handlers.features.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        mock_plugin_runner._validate_requirements.return_value = (False, ["missing-lib"])

        with patch.object(mod, "get_registry", return_value=mock_registry):
            result = plugins_handler.handle("/api/v1/plugins/test-plugin-1", {}, None)

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
        result = plugins_handler.handle("/api/v1/plugins/..%2F..%2Fetc", {}, None)
        assert result.status_code == 400

    def test_sql_injection_blocked(self, plugins_handler):
        result = plugins_handler.handle("/api/v1/plugins/'; DROP TABLE plugins;--", {}, None)
        assert result.status_code == 400

    def test_xss_blocked(self, plugins_handler):
        result = plugins_handler.handle("/api/v1/plugins/<script>alert(1)</script>", {}, None)
        # Path may not match at all (None) or should be rejected (400)
        assert result is None or result.status_code == 400

    def test_valid_plugin_names_accepted(self, plugins_handler, mock_registry):
        import aragora.server.handlers.features.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        valid_names = ["test-plugin", "plugin_name", "plugin123"]
        for name in valid_names:
            with patch.object(mod, "get_registry", return_value=mock_registry):
                result = plugins_handler.handle(f"/api/v1/plugins/{name}", {}, None)
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
        import aragora.server.handlers.features.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        mock_registry = Mock()
        mock_registry.list_plugins.side_effect = Exception("Database error")

        with patch.object(mod, "get_registry", return_value=mock_registry):
            result = plugins_handler.handle("/api/v1/plugins", {}, None)

            assert result is not None
            assert result.status_code == 500

    def test_get_plugin_exception(self, plugins_handler):
        import aragora.server.handlers.features.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        mock_registry = Mock()
        mock_registry.get.side_effect = Exception("Database error")

        with patch.object(mod, "get_registry", return_value=mock_registry):
            result = plugins_handler.handle("/api/v1/plugins/test", {}, None)

            assert result is not None
            assert result.status_code == 500


# ============================================================================
# Edge Cases
# ============================================================================


class TestPluginsEdgeCases:
    """Tests for edge cases."""

    def test_empty_plugin_name(self, plugins_handler):
        # Empty plugin name shouldn't match
        result = plugins_handler.handle("/api/v1/plugins/", {}, None)
        # Should not be handled (None or 400)
        assert result is None or result.status_code == 400

    def test_very_long_plugin_name(self, plugins_handler):
        long_name = "a" * 1000
        result = plugins_handler.handle(f"/api/v1/plugins/{long_name}", {}, None)
        # Should handle gracefully
        assert result is not None

    def test_plugin_without_runner(self, plugins_handler, mock_registry, mock_plugin_manifest):
        import aragora.server.handlers.features.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        # No runner available
        mock_registry.get_runner.return_value = None

        with patch.object(mod, "get_registry", return_value=mock_registry):
            result = plugins_handler.handle("/api/v1/plugins/test-plugin-1", {}, None)

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            # Should return manifest without requirements info
            assert data["name"] == "test-plugin-1"
            assert "requirements_satisfied" not in data


# ============================================================================
# Path Variant Tests (Legacy vs Versioned)
# ============================================================================


class TestPluginPathVariants:
    """Tests for legacy and versioned path variants.

    Both /api/plugins/* (legacy) and /api/v1/plugins/* (versioned) should work
    identically, but legacy paths should include Sunset/Deprecation headers.
    """

    def test_can_handle_legacy_plugins_list(self, plugins_handler):
        """Legacy /api/plugins path should be handled."""
        assert plugins_handler.can_handle("/api/plugins") is True

    def test_can_handle_versioned_plugins_list(self, plugins_handler):
        """Versioned /api/v1/plugins path should be handled."""
        assert plugins_handler.can_handle("/api/v1/plugins") is True

    def test_can_handle_legacy_plugin_details(self, plugins_handler):
        """Legacy /api/plugins/{name} path should be handled."""
        assert plugins_handler.can_handle("/api/plugins/test-plugin") is True

    def test_can_handle_legacy_plugin_run(self, plugins_handler):
        """Legacy /api/plugins/{name}/run path should be handled."""
        assert plugins_handler.can_handle("/api/plugins/test/run") is True

    def test_can_handle_legacy_plugin_install(self, plugins_handler):
        """Legacy /api/plugins/{name}/install path should be handled."""
        assert plugins_handler.can_handle("/api/plugins/test/install") is True

    @pytest.mark.parametrize("path_prefix", ["/api/plugins", "/api/v1/plugins"])
    def test_list_plugins_both_paths(self, plugins_handler, mock_registry, path_prefix):
        """Both path variants return identical plugin list."""
        import aragora.server.handlers.features.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        with patch.object(mod, "get_registry", return_value=mock_registry):
            result = plugins_handler.handle(path_prefix, {}, None)

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert "plugins" in data
            assert data["count"] == 2

    @pytest.mark.parametrize("path_prefix", ["/api/plugins", "/api/v1/plugins"])
    def test_get_plugin_both_paths(self, plugins_handler, mock_registry, path_prefix):
        """Both path variants return identical plugin details."""
        import aragora.server.handlers.features.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        with patch.object(mod, "get_registry", return_value=mock_registry):
            result = plugins_handler.handle(f"{path_prefix}/test-plugin-1", {}, None)

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["name"] == "test-plugin-1"


# ============================================================================
# Sunset Header Tests
# ============================================================================


class TestSunsetHeaders:
    """Tests for HTTP Sunset and Deprecation headers on legacy paths.

    Per RFC 8594, the Sunset header indicates when an API endpoint will be retired.
    Legacy /api/plugins/* paths should include these headers.
    """

    def test_legacy_path_has_sunset_header(self, plugins_handler, mock_registry):
        """Legacy paths should include Sunset header."""
        import aragora.server.handlers.features.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        with patch.object(mod, "get_registry", return_value=mock_registry):
            result = plugins_handler.handle("/api/plugins", {}, None)

            assert result is not None
            assert result.status_code == 200
            assert result.headers is not None
            assert "Sunset" in result.headers
            assert "2026" in result.headers["Sunset"]

    def test_legacy_path_has_deprecation_header(self, plugins_handler, mock_registry):
        """Legacy paths should include Deprecation header."""
        import aragora.server.handlers.features.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        with patch.object(mod, "get_registry", return_value=mock_registry):
            result = plugins_handler.handle("/api/plugins", {}, None)

            assert result is not None
            assert result.headers is not None
            assert "Deprecation" in result.headers
            assert result.headers["Deprecation"] == "true"

    def test_versioned_path_no_sunset_header(self, plugins_handler, mock_registry):
        """Versioned paths should NOT include Sunset header."""
        import aragora.server.handlers.features.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        with patch.object(mod, "get_registry", return_value=mock_registry):
            result = plugins_handler.handle("/api/v1/plugins", {}, None)

            assert result is not None
            assert result.status_code == 200
            # Headers may be None or not contain Sunset
            if result.headers:
                assert "Sunset" not in result.headers

    def test_legacy_plugin_details_has_sunset(self, plugins_handler, mock_registry):
        """Legacy plugin details path should include Sunset header."""
        import aragora.server.handlers.features.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        with patch.object(mod, "get_registry", return_value=mock_registry):
            result = plugins_handler.handle("/api/plugins/test-plugin-1", {}, None)

            assert result is not None
            assert result.headers is not None
            assert "Sunset" in result.headers

    def test_versioned_plugin_details_no_sunset(self, plugins_handler, mock_registry):
        """Versioned plugin details path should NOT include Sunset header."""
        import aragora.server.handlers.features.plugins as mod

        if not mod.PLUGINS_AVAILABLE:
            pytest.skip("Plugins module not available")

        with patch.object(mod, "get_registry", return_value=mock_registry):
            result = plugins_handler.handle("/api/v1/plugins/test-plugin-1", {}, None)

            assert result is not None
            if result.headers:
                assert "Sunset" not in result.headers


# ============================================================================
# Plugin Submission Security Tests
# ============================================================================


class TestPluginSubmissionSecurity:
    """Security tests for plugin submission endpoint (POST /api/v1/plugins/submit).

    Tests validate that the submission handler properly enforces:
    - Name length limits (max 64 chars)
    - Name format (lowercase, starts with letter)
    - Entry point format (module.path:function)
    - Version format (semver)
    - Schema validation via PLUGIN_MANIFEST_SCHEMA
    """

    _test_counter = 0

    def _submit_plugin(self, plugins_handler, manifest_data, user_id=None):
        """Helper to submit a plugin with the given manifest."""
        import uuid

        body = {"manifest": manifest_data}

        # Use unique user ID and IP for each test to avoid rate limits
        TestPluginSubmissionSecurity._test_counter += 1
        unique_id = f"test-user-{uuid.uuid4().hex[:8]}"
        unique_ip = f"192.168.{(self._test_counter // 256) % 256}.{self._test_counter % 256}"

        if user_id is None:
            user_id = unique_id

        # Create mock handler with auth headers and unique client address
        mock_handler = Mock()
        mock_handler.headers = {
            "Authorization": "Bearer test-api-token-12345",
            "Content-Type": "application/json",
        }
        mock_handler.client_address = (unique_ip, 12345)

        # Patch auth_config to accept our test token
        from aragora.server.auth import auth_config

        with (
            patch.object(auth_config, "api_token", "test-api-token-12345"),
            patch.object(auth_config, "validate_token", return_value=True),
            patch.object(plugins_handler, "get_user_id", return_value=user_id),
            patch.object(plugins_handler, "read_json_body", return_value=body),
        ):
            # Use handle_post for POST requests with the mock handler
            return plugins_handler.handle_post("/api/v1/plugins/submit", {}, mock_handler)

    def test_submit_plugin_name_too_long(self, plugins_handler):
        """Reject plugin names exceeding 64 characters."""
        manifest = {
            "name": "a" * 65,  # 65 chars, exceeds limit
            "version": "1.0.0",
            "description": "A test plugin",
            "entry_point": "my_plugin:main",
        }
        result = self._submit_plugin(plugins_handler, manifest)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "error" in data

    def test_submit_plugin_name_at_max_length(self, plugins_handler):
        """Accept plugin names at exactly 64 characters."""
        import aragora.server.handlers.features.plugins as mod

        # Skip if plugins not available
        original_available = mod.PLUGINS_AVAILABLE
        mod.PLUGINS_AVAILABLE = False  # Disable manifest.validate() check

        try:
            manifest = {
                "name": "a" + "b" * 62 + "c",  # 64 chars: starts with letter, ends with letter
                "version": "1.0.0",
                "description": "A test plugin",
                "entry_point": "my_plugin:main",
            }
            result = self._submit_plugin(plugins_handler, manifest)

            # Should pass schema validation (name length OK)
            # May fail on other checks but not on name length
            if result.status_code == 400:
                data = json.loads(result.body)
                # Should not fail on name length
                assert "max_length" not in data.get("error", "").lower()
        finally:
            mod.PLUGINS_AVAILABLE = original_available

    def test_submit_plugin_name_starts_with_number(self, plugins_handler):
        """Reject plugin names starting with a number."""
        manifest = {
            "name": "1plugin",  # Starts with number
            "version": "1.0.0",
            "description": "A test plugin",
            "entry_point": "my_plugin:main",
        }
        result = self._submit_plugin(plugins_handler, manifest)

        assert result is not None
        assert result.status_code == 400

    def test_submit_plugin_name_uppercase_rejected(self, plugins_handler):
        """Reject plugin names with uppercase letters."""
        manifest = {
            "name": "MyPlugin",  # Has uppercase
            "version": "1.0.0",
            "description": "A test plugin",
            "entry_point": "my_plugin:main",
        }
        result = self._submit_plugin(plugins_handler, manifest)

        assert result is not None
        assert result.status_code == 400

    def test_submit_plugin_name_special_chars_rejected(self, plugins_handler):
        """Reject plugin names with special characters."""
        invalid_names = [
            "my_plugin",  # Underscore not allowed
            "my.plugin",  # Dot not allowed
            "my plugin",  # Space not allowed
            "my@plugin",  # @ not allowed
            "../evil",  # Path traversal
        ]
        for name in invalid_names:
            manifest = {
                "name": name,
                "version": "1.0.0",
                "description": "A test plugin",
                "entry_point": "my_plugin:main",
            }
            result = self._submit_plugin(plugins_handler, manifest)

            assert result is not None, f"Name '{name}' should be rejected"
            assert result.status_code == 400, f"Name '{name}' should return 400"

    def test_submit_plugin_entry_point_invalid_format(self, plugins_handler):
        """Reject invalid entry_point format (must be module:function)."""
        invalid_entry_points = [
            "no_colon_here",  # Missing colon
            ":function",  # Missing module
            "module:",  # Missing function
            "module:func:extra",  # Extra colon
            "../../evil:run",  # Path traversal attempt
            "123module:func",  # Module starts with number
        ]
        for entry_point in invalid_entry_points:
            manifest = {
                "name": "test-plugin",
                "version": "1.0.0",
                "description": "A test plugin",
                "entry_point": entry_point,
            }
            result = self._submit_plugin(plugins_handler, manifest)

            assert result is not None, f"Entry point '{entry_point}' should be rejected"
            assert result.status_code == 400, f"Entry point '{entry_point}' should return 400"

    def test_submit_plugin_entry_point_valid_formats(self, plugins_handler):
        """Accept valid entry_point formats."""
        import aragora.server.handlers.features.plugins as mod

        original_available = mod.PLUGINS_AVAILABLE
        mod.PLUGINS_AVAILABLE = False  # Disable manifest.validate() check

        try:
            valid_entry_points = [
                "module:function",
                "my_module:my_function",
                "package.module:function",
                "deep.nested.module:handler",
                "_private:_handler",
            ]
            for entry_point in valid_entry_points:
                manifest = {
                    "name": "test-plugin",
                    "version": "1.0.0",
                    "description": "A test plugin",
                    "entry_point": entry_point,
                }
                result = self._submit_plugin(plugins_handler, manifest)

                # Should not fail on entry_point validation
                if result.status_code == 400:
                    data = json.loads(result.body)
                    assert (
                        "entry_point" not in data.get("error", "").lower()
                    ), f"Entry point '{entry_point}' should be valid"
        finally:
            mod.PLUGINS_AVAILABLE = original_available

    def test_submit_plugin_version_invalid_format(self, plugins_handler):
        """Reject invalid version formats (must be semver)."""
        invalid_versions = [
            "1.0",  # Missing patch
            "v1.0.0",  # Leading 'v'
            "1.0.0.0",  # Extra component
            "1.a.0",  # Non-numeric
            "latest",  # Not semver
            "",  # Empty
        ]
        for version in invalid_versions:
            manifest = {
                "name": "test-plugin",
                "version": version,
                "description": "A test plugin",
                "entry_point": "my_plugin:main",
            }
            result = self._submit_plugin(plugins_handler, manifest)

            assert result is not None, f"Version '{version}' should be rejected"
            assert result.status_code == 400, f"Version '{version}' should return 400"

    def test_submit_plugin_version_valid_semver(self, plugins_handler):
        """Accept valid semver versions."""
        import aragora.server.handlers.features.plugins as mod

        original_available = mod.PLUGINS_AVAILABLE
        mod.PLUGINS_AVAILABLE = False

        try:
            valid_versions = [
                "1.0.0",
                "0.1.0",
                "10.20.30",
                "1.0.0-alpha",
                "1.0.0-beta.1",
                "1.0.0+build.123",
            ]
            for version in valid_versions:
                manifest = {
                    "name": "test-plugin",
                    "version": version,
                    "description": "A test plugin",
                    "entry_point": "my_plugin:main",
                }
                result = self._submit_plugin(plugins_handler, manifest)

                if result.status_code == 400:
                    data = json.loads(result.body)
                    assert (
                        "version" not in data.get("error", "").lower()
                    ), f"Version '{version}' should be valid"
        finally:
            mod.PLUGINS_AVAILABLE = original_available

    def test_submit_plugin_description_too_long(self, plugins_handler):
        """Reject descriptions exceeding 1000 characters."""
        manifest = {
            "name": "test-plugin",
            "version": "1.0.0",
            "description": "x" * 1001,  # Exceeds limit
            "entry_point": "my_plugin:main",
        }
        result = self._submit_plugin(plugins_handler, manifest)

        assert result is not None
        assert result.status_code == 400

    def test_submit_plugin_missing_required_fields(self, plugins_handler):
        """Reject submissions missing required manifest fields."""
        # Missing name
        result = self._submit_plugin(
            plugins_handler,
            {
                "version": "1.0.0",
                "description": "Test",
                "entry_point": "mod:func",
            },
        )
        assert result.status_code == 400

        # Missing version
        result = self._submit_plugin(
            plugins_handler,
            {
                "name": "test-plugin",
                "description": "Test",
                "entry_point": "mod:func",
            },
        )
        assert result.status_code == 400

        # Missing entry_point
        result = self._submit_plugin(
            plugins_handler,
            {
                "name": "test-plugin",
                "version": "1.0.0",
                "description": "Test",
            },
        )
        assert result.status_code == 400

    def test_submit_plugin_valid_manifest(self, plugins_handler):
        """Accept a fully valid manifest submission."""
        import aragora.server.handlers.features.plugins as mod

        original_available = mod.PLUGINS_AVAILABLE
        mod.PLUGINS_AVAILABLE = False

        try:
            manifest = {
                "name": "my-awesome-plugin",
                "version": "1.0.0",
                "description": "An awesome plugin for testing",
                "entry_point": "my_plugin.main:handler",
                "author": "Test Author",
                "category": "analysis",
            }
            result = self._submit_plugin(plugins_handler, manifest)

            # Should succeed (200) or conflict (409 if already submitted)
            assert result is not None
            assert result.status_code in [200, 409]
        finally:
            mod.PLUGINS_AVAILABLE = original_available

    def test_submit_plugin_category_invalid(self, plugins_handler):
        """Reject invalid category values."""
        manifest = {
            "name": "test-plugin",
            "version": "1.0.0",
            "description": "A test plugin",
            "entry_point": "my_plugin:main",
            "category": "invalid-category",  # Not in allowed values
        }
        result = self._submit_plugin(plugins_handler, manifest)

        assert result is not None
        assert result.status_code == 400
