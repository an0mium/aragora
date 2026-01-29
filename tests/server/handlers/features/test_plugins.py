"""Tests for Plugins Handler."""

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import pytest
from unittest.mock import MagicMock, patch

from aragora.server.handlers.features.plugins import (
    PluginsHandler,
    _installed_plugins,
    _plugin_submissions,
    PLUGINS_AVAILABLE,
)


@pytest.fixture(autouse=True)
def clear_plugin_stores():
    """Clear plugin stores between tests."""
    _installed_plugins.clear()
    _plugin_submissions.clear()
    yield


@pytest.fixture
def handler():
    """Create handler instance."""
    return PluginsHandler(ctx={})


class TestPluginsHandler:
    """Tests for PluginsHandler class."""

    def test_handler_creation(self, handler):
        """Test creating handler instance."""
        assert handler is not None

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(PluginsHandler, "ROUTES")
        routes = PluginsHandler.ROUTES
        assert "/api/v1/plugins" in routes
        assert "/api/v1/plugins/installed" in routes
        assert "/api/v1/plugins/marketplace" in routes
        assert "/api/v1/plugins/submit" in routes
        assert "/api/v1/plugins/submissions" in routes

    def test_can_handle_base_routes(self, handler):
        """Test can_handle for base routes."""
        assert handler.can_handle("/api/v1/plugins") is True
        assert handler.can_handle("/api/v1/plugins/installed") is True
        assert handler.can_handle("/api/v1/plugins/marketplace") is True

    def test_can_handle_plugin_routes(self, handler):
        """Test can_handle for plugin-specific routes."""
        assert handler.can_handle("/api/v1/plugins/myplugin") is True
        assert handler.can_handle("/api/v1/plugins/myplugin/run") is True
        assert handler.can_handle("/api/v1/plugins/myplugin/install") is True

    def test_can_handle_legacy_routes(self, handler):
        """Test can_handle for legacy routes."""
        assert handler.can_handle("/api/plugins") is True
        assert handler.can_handle("/api/plugins/installed") is True
        assert handler.can_handle("/api/plugins/marketplace") is True

    def test_can_handle_invalid_routes(self, handler):
        """Test can_handle rejects invalid routes."""
        assert handler.can_handle("/api/v1/extensions/") is False
        assert handler.can_handle("/api/v1/invalid/route") is False


class TestPluginsList:
    """Tests for list plugins endpoint."""

    def test_list_plugins_available(self, handler):
        """Test listing plugins when module available."""
        mock_handler = MagicMock()
        mock_handler.path = "/api/v1/plugins"

        with patch(
            "aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True
        ), patch("aragora.server.handlers.features.plugins.get_registry") as mock_get_reg:
            mock_registry = MagicMock()
            mock_plugin = MagicMock()
            mock_plugin.to_dict.return_value = {"name": "test-plugin", "version": "1.0.0"}
            mock_registry.list_plugins.return_value = [mock_plugin]
            mock_get_reg.return_value = mock_registry

            result = handler.handle("/api/v1/plugins", {}, mock_handler)
            assert result.status == 200

    def test_list_plugins_unavailable(self, handler):
        """Test listing plugins when module unavailable."""
        mock_handler = MagicMock()
        mock_handler.path = "/api/v1/plugins"

        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", False):
            result = handler.handle("/api/v1/plugins", {}, mock_handler)
            assert result.status == 503


class TestPluginsMarketplace:
    """Tests for marketplace endpoint."""

    def test_get_marketplace_available(self, handler):
        """Test getting marketplace when module available."""
        mock_handler = MagicMock()
        mock_handler.path = "/api/v1/plugins/marketplace"

        with patch(
            "aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True
        ), patch("aragora.server.handlers.features.plugins.get_registry") as mock_get_reg:
            mock_registry = MagicMock()
            mock_plugin = MagicMock()
            mock_plugin.to_dict.return_value = {"name": "test-plugin"}
            mock_plugin.featured = False
            mock_plugin.category = "utility"
            mock_registry.list_plugins.return_value = [mock_plugin]
            mock_get_reg.return_value = mock_registry

            result = handler.handle("/api/v1/plugins/marketplace", {}, mock_handler)
            assert result.status == 200

            import json

            body = json.loads(result.body)
            assert "featured" in body
            assert "categories" in body
            assert "total" in body

    def test_get_marketplace_unavailable(self, handler):
        """Test getting marketplace when module unavailable."""
        mock_handler = MagicMock()
        mock_handler.path = "/api/v1/plugins/marketplace"

        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", False):
            result = handler.handle("/api/v1/plugins/marketplace", {}, mock_handler)
            assert result.status == 200

            import json

            body = json.loads(result.body)
            assert body["total"] == 0
            assert "Plugin system not configured" in body.get("message", "")


class TestPluginsInstalled:
    """Tests for installed plugins endpoint."""

    def test_list_installed_unauthenticated(self, handler):
        """Test listing installed requires auth."""
        mock_handler = MagicMock()
        mock_handler.path = "/api/v1/plugins/installed"

        with patch.object(handler, "get_current_user", return_value=None):
            result = handler.handle("/api/v1/plugins/installed", {}, mock_handler)
            assert result.status == 401

    def test_list_installed_empty(self, handler):
        """Test listing installed with no plugins."""
        mock_handler = MagicMock()
        mock_handler.path = "/api/v1/plugins/installed"
        mock_user = MagicMock()
        mock_user.user_id = "user123"

        with patch.object(handler, "get_current_user", return_value=mock_user):
            result = handler.handle("/api/v1/plugins/installed", {}, mock_handler)
            assert result.status == 200

            import json

            body = json.loads(result.body)
            assert body["count"] == 0
            assert body["installed"] == []


class TestPluginDetails:
    """Tests for plugin details endpoint."""

    def test_get_plugin_not_found(self, handler):
        """Test getting non-existent plugin."""
        mock_handler = MagicMock()
        mock_handler.path = "/api/v1/plugins/nonexistent"

        with patch(
            "aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True
        ), patch("aragora.server.handlers.features.plugins.get_registry") as mock_get_reg:
            mock_registry = MagicMock()
            mock_registry.get.return_value = None
            mock_get_reg.return_value = mock_registry

            result = handler.handle("/api/v1/plugins/nonexistent", {}, mock_handler)
            assert result.status == 404

    def test_get_plugin_success(self, handler):
        """Test getting existing plugin."""
        mock_handler = MagicMock()
        mock_handler.path = "/api/v1/plugins/test-plugin"

        with patch(
            "aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True
        ), patch("aragora.server.handlers.features.plugins.get_registry") as mock_get_reg:
            mock_registry = MagicMock()
            mock_manifest = MagicMock()
            mock_manifest.to_dict.return_value = {"name": "test-plugin", "version": "1.0.0"}
            mock_registry.get.return_value = mock_manifest
            mock_registry.get_runner.return_value = None
            mock_get_reg.return_value = mock_registry

            result = handler.handle("/api/v1/plugins/test-plugin", {}, mock_handler)
            assert result.status == 200


class TestPluginInstall:
    """Tests for plugin installation."""

    def test_install_plugin_unauthenticated(self, handler):
        """Test install requires auth."""
        mock_handler = MagicMock()
        mock_handler.path = "/api/v1/plugins/test-plugin/install"

        with (
            patch.object(handler, "get_current_user", return_value=None),
            patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True),
        ):
            result = handler.handle_post(
                "/api/v1/plugins/test-plugin/install", {}, mock_handler
            )
            assert result.status == 401

    def test_install_plugin_not_found(self, handler):
        """Test install non-existent plugin."""
        mock_handler = MagicMock()
        mock_handler.path = "/api/v1/plugins/nonexistent/install"
        mock_user = MagicMock()
        mock_user.user_id = "user123"

        with (
            patch.object(handler, "get_current_user", return_value=mock_user),
            patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True),
            patch("aragora.server.handlers.features.plugins.get_registry") as mock_get_reg,
            patch("aragora.server.handlers.features.plugins.require_permission", lambda p: lambda f: f),
            patch("aragora.server.handlers.features.plugins.rate_limit", lambda **kw: lambda f: f),
        ):
            mock_registry = MagicMock()
            mock_registry.get.return_value = None
            mock_get_reg.return_value = mock_registry

            result = handler._install_plugin("nonexistent", mock_handler)
            assert result.status == 404


class TestPluginUninstall:
    """Tests for plugin uninstallation."""

    def test_uninstall_plugin_not_installed(self, handler):
        """Test uninstall non-installed plugin."""
        mock_handler = MagicMock()
        mock_handler.path = "/api/v1/plugins/test-plugin/install"
        mock_user = MagicMock()
        mock_user.user_id = "user123"

        with patch.object(handler, "get_current_user", return_value=mock_user):
            result = handler._uninstall_plugin("test-plugin", mock_handler)
            assert result.status == 404

    def test_uninstall_plugin_success(self, handler):
        """Test successful plugin uninstallation."""
        mock_handler = MagicMock()
        mock_handler.path = "/api/v1/plugins/test-plugin/install"
        mock_user = MagicMock()
        mock_user.user_id = "user123"

        # Pre-install the plugin
        _installed_plugins["user123"] = {"test-plugin": {"installed_at": "2024-01-01"}}

        with patch.object(handler, "get_current_user", return_value=mock_user):
            result = handler._uninstall_plugin("test-plugin", mock_handler)
            assert result.status == 200
            assert "test-plugin" not in _installed_plugins.get("user123", {})


class TestPluginRun:
    """Tests for plugin execution."""

    def test_run_plugin_unavailable(self, handler):
        """Test run when module unavailable."""
        mock_handler = MagicMock()
        mock_handler.path = "/api/v1/plugins/test-plugin/run"

        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", False):
            result = handler._run_plugin("test-plugin", mock_handler)
            assert result.status == 503

    def test_run_plugin_invalid_body(self, handler):
        """Test run with invalid JSON body."""
        mock_handler = MagicMock()
        mock_handler.path = "/api/v1/plugins/test-plugin/run"

        with (
            patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True),
            patch.object(handler, "read_json_body", return_value=None),
        ):
            result = handler._run_plugin("test-plugin", mock_handler)
            assert result.status == 400


class TestPluginSubmit:
    """Tests for plugin submission."""

    def test_submit_plugin_unauthenticated(self, handler):
        """Test submit requires auth."""
        mock_handler = MagicMock()
        mock_handler.path = "/api/v1/plugins/submit"

        with patch.object(handler, "get_current_user", return_value=None):
            result = handler._submit_plugin(mock_handler)
            assert result.status == 401

    def test_submit_plugin_missing_manifest(self, handler):
        """Test submit requires manifest."""
        mock_handler = MagicMock()
        mock_handler.path = "/api/v1/plugins/submit"
        mock_user = MagicMock()
        mock_user.user_id = "user123"

        with (
            patch.object(handler, "get_current_user", return_value=mock_user),
            patch.object(handler, "read_json_body", return_value={}),
        ):
            result = handler._submit_plugin(mock_handler)
            assert result.status == 400


class TestPluginSubmissions:
    """Tests for listing plugin submissions."""

    def test_list_submissions_unauthenticated(self, handler):
        """Test list submissions requires auth."""
        mock_handler = MagicMock()
        mock_handler.path = "/api/v1/plugins/submissions"

        with patch.object(handler, "get_current_user", return_value=None):
            result = handler.handle("/api/v1/plugins/submissions", {}, mock_handler)
            assert result.status == 401

    def test_list_submissions_empty(self, handler):
        """Test list submissions when none exist."""
        mock_handler = MagicMock()
        mock_handler.path = "/api/v1/plugins/submissions"
        mock_user = MagicMock()
        mock_user.user_id = "user123"

        with patch.object(handler, "get_current_user", return_value=mock_user):
            result = handler.handle("/api/v1/plugins/submissions", {}, mock_handler)
            assert result.status == 200

            import json

            body = json.loads(result.body)
            assert body["count"] == 0
            assert body["submissions"] == []


class TestLegacyPathHandling:
    """Tests for legacy path deprecation."""

    def test_is_legacy_path(self, handler):
        """Test legacy path detection."""
        assert handler._is_legacy_path("/api/plugins") is True
        assert handler._is_legacy_path("/api/plugins/test") is True
        assert handler._is_legacy_path("/api/v1/plugins") is False
        assert handler._is_legacy_path("/api/v1/plugins/test") is False

    def test_normalize_plugin_path(self, handler):
        """Test path normalization."""
        assert handler._normalize_plugin_path("/api/v1/plugins/test") == "/api/plugins/test"
        assert handler._normalize_plugin_path("/api/plugins/test") == "/api/plugins/test"

    def test_sunset_header_added_for_legacy(self, handler):
        """Test Sunset header is added for legacy paths."""
        from aragora.server.handlers.base import HandlerResult

        response = HandlerResult(status=200, body="{}")
        mock_handler = MagicMock()
        mock_handler.path = "/api/plugins"

        result = handler._add_sunset_header_if_legacy("/api/plugins", response, mock_handler)
        assert result.headers is not None
        assert "Sunset" in result.headers
        assert result.headers["Deprecation"] == "true"

    def test_no_sunset_header_for_versioned(self, handler):
        """Test no Sunset header for versioned paths."""
        from aragora.server.handlers.base import HandlerResult

        response = HandlerResult(status=200, body="{}")
        mock_handler = MagicMock()
        mock_handler.path = "/api/v1/plugins"

        result = handler._add_sunset_header_if_legacy("/api/v1/plugins", response, mock_handler)
        assert result.headers is None or "Sunset" not in result.headers
