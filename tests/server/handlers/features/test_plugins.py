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
    _plugins_limiter,
)


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter between tests."""
    _plugins_limiter._buckets.clear()
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
        assert "/api/v1/plugins/available" in routes
        assert "/api/v1/plugins/installed" in routes
        assert "/api/v1/plugins/refresh" in routes

    def test_can_handle_base_routes(self, handler):
        """Test can_handle for base routes."""
        assert handler.can_handle("/api/v1/plugins") is True
        assert handler.can_handle("/api/v1/plugins/available") is True
        assert handler.can_handle("/api/v1/plugins/installed") is True

    def test_can_handle_plugin_routes(self, handler):
        """Test can_handle for plugin-specific routes."""
        assert handler.can_handle("/api/v1/plugins/myplugin") is True
        assert handler.can_handle("/api/v1/plugins/myplugin/enable") is True
        assert handler.can_handle("/api/v1/plugins/myplugin/disable") is True
        assert handler.can_handle("/api/v1/plugins/myplugin/settings") is True
        assert handler.can_handle("/api/v1/plugins/myplugin/uninstall") is True

    def test_can_handle_legacy_routes(self, handler):
        """Test can_handle for legacy routes."""
        assert handler.can_handle("/plugins/available") is True
        assert handler.can_handle("/plugins/installed") is True

    def test_can_handle_invalid_routes(self, handler):
        """Test can_handle rejects invalid routes."""
        assert handler.can_handle("/api/v1/extensions/") is False
        assert handler.can_handle("/api/v1/invalid/route") is False


class TestPluginsAvailable:
    """Tests for available plugins endpoint."""

    def test_get_available_plugins(self, handler):
        """Test listing available plugins."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.features.plugins.get_plugin_registry") as mock_registry:
            mock_registry.return_value = MagicMock(
                list_available=lambda: [
                    MagicMock(
                        id="test-plugin",
                        name="Test Plugin",
                        version="1.0.0",
                        description="A test plugin",
                        to_dict=lambda: {"id": "test-plugin"},
                    )
                ]
            )

            result = handler.handle("/api/v1/plugins/available", {}, mock_handler)
            assert result.status == 200

    def test_get_available_plugins_registry_error(self, handler):
        """Test available plugins when registry unavailable."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.features.plugins.get_plugin_registry") as mock_registry:
            mock_registry.side_effect = ImportError("Registry not available")

            result = handler.handle("/api/v1/plugins/available", {}, mock_handler)
            assert result.status == 200  # Returns empty list


class TestPluginsInstalled:
    """Tests for installed plugins endpoint."""

    def test_get_installed_plugins(self, handler):
        """Test listing installed plugins."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.features.plugins.get_plugin_registry") as mock_registry:
            mock_registry.return_value = MagicMock(
                list_installed=lambda: [
                    MagicMock(
                        id="test-plugin",
                        name="Test Plugin",
                        enabled=True,
                        to_dict=lambda: {"id": "test-plugin"},
                    )
                ]
            )

            result = handler.handle("/api/v1/plugins/installed", {}, mock_handler)
            assert result.status == 200


class TestPluginsInstall:
    """Tests for plugin installation."""

    def test_install_plugin_missing_plugin_id(self, handler):
        """Test install requires plugin_id."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch.object(handler, "read_json_body", return_value={}):
            result = handler.handle_post("/api/v1/plugins", {}, mock_handler)
            assert result.status == 400

    def test_install_plugin_success(self, handler):
        """Test successful plugin installation."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with (
            patch.object(handler, "read_json_body", return_value={"plugin_id": "test-plugin"}),
            patch("aragora.server.handlers.features.plugins.get_plugin_registry") as mock_registry,
        ):
            mock_reg = MagicMock()
            mock_reg.install.return_value = MagicMock(
                success=True, plugin=MagicMock(to_dict=lambda: {"id": "test-plugin"})
            )
            mock_registry.return_value = mock_reg

            result = handler.handle_post("/api/v1/plugins", {}, mock_handler)
            assert result.status == 200


class TestPluginsUninstall:
    """Tests for plugin uninstallation."""

    def test_uninstall_plugin_not_found(self, handler):
        """Test uninstall non-existent plugin."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.features.plugins.get_plugin_registry") as mock_registry:
            mock_reg = MagicMock()
            mock_reg.get_installed.return_value = None
            mock_registry.return_value = mock_reg

            result = handler.handle_delete(
                "/api/v1/plugins/invalid-plugin/uninstall", {}, mock_handler
            )
            assert result.status == 404

    def test_uninstall_plugin_success(self, handler):
        """Test successful plugin uninstallation."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.features.plugins.get_plugin_registry") as mock_registry:
            mock_reg = MagicMock()
            mock_reg.get_installed.return_value = MagicMock(id="test-plugin")
            mock_reg.uninstall.return_value = True
            mock_registry.return_value = mock_reg

            result = handler.handle_delete(
                "/api/v1/plugins/test-plugin/uninstall", {}, mock_handler
            )
            assert result.status == 200


class TestPluginsEnable:
    """Tests for plugin enable/disable."""

    def test_enable_plugin_not_found(self, handler):
        """Test enable non-existent plugin."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.features.plugins.get_plugin_registry") as mock_registry:
            mock_reg = MagicMock()
            mock_reg.get_installed.return_value = None
            mock_registry.return_value = mock_reg

            result = handler.handle_post("/api/v1/plugins/invalid-plugin/enable", {}, mock_handler)
            assert result.status == 404

    def test_enable_plugin_success(self, handler):
        """Test successful plugin enable."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.features.plugins.get_plugin_registry") as mock_registry:
            mock_reg = MagicMock()
            mock_plugin = MagicMock(id="test-plugin", enabled=False)
            mock_reg.get_installed.return_value = mock_plugin
            mock_reg.enable.return_value = True
            mock_registry.return_value = mock_reg

            result = handler.handle_post("/api/v1/plugins/test-plugin/enable", {}, mock_handler)
            assert result.status == 200

    def test_disable_plugin_success(self, handler):
        """Test successful plugin disable."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.features.plugins.get_plugin_registry") as mock_registry:
            mock_reg = MagicMock()
            mock_plugin = MagicMock(id="test-plugin", enabled=True)
            mock_reg.get_installed.return_value = mock_plugin
            mock_reg.disable.return_value = True
            mock_registry.return_value = mock_reg

            result = handler.handle_post("/api/v1/plugins/test-plugin/disable", {}, mock_handler)
            assert result.status == 200


class TestPluginsSettings:
    """Tests for plugin settings."""

    def test_get_plugin_settings(self, handler):
        """Test getting plugin settings."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.features.plugins.get_plugin_registry") as mock_registry:
            mock_reg = MagicMock()
            mock_plugin = MagicMock(
                id="test-plugin",
                settings={"key": "value"},
                settings_schema={"type": "object"},
            )
            mock_reg.get_installed.return_value = mock_plugin
            mock_registry.return_value = mock_reg

            result = handler.handle("/api/v1/plugins/test-plugin/settings", {}, mock_handler)
            assert result.status == 200

    def test_update_plugin_settings(self, handler):
        """Test updating plugin settings."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with (
            patch.object(handler, "read_json_body", return_value={"key": "new_value"}),
            patch("aragora.server.handlers.features.plugins.get_plugin_registry") as mock_registry,
        ):
            mock_reg = MagicMock()
            mock_plugin = MagicMock(id="test-plugin", settings={})
            mock_reg.get_installed.return_value = mock_plugin
            mock_reg.update_settings.return_value = True
            mock_registry.return_value = mock_reg

            result = handler.handle_post("/api/v1/plugins/test-plugin/settings", {}, mock_handler)
            assert result.status == 200

    def test_get_plugin_settings_not_found(self, handler):
        """Test getting settings for non-existent plugin."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.features.plugins.get_plugin_registry") as mock_registry:
            mock_reg = MagicMock()
            mock_reg.get_installed.return_value = None
            mock_registry.return_value = mock_reg

            result = handler.handle("/api/v1/plugins/invalid-plugin/settings", {}, mock_handler)
            assert result.status == 404


class TestPluginsRefresh:
    """Tests for plugin refresh."""

    def test_refresh_plugins(self, handler):
        """Test refreshing plugin registry."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.features.plugins.get_plugin_registry") as mock_registry:
            mock_reg = MagicMock()
            mock_reg.refresh.return_value = True
            mock_reg.list_available.return_value = []
            mock_registry.return_value = mock_reg

            result = handler.handle_post("/api/v1/plugins/refresh", {}, mock_handler)
            assert result.status == 200


class TestPluginsRateLimiting:
    """Tests for plugin rate limiting."""

    def test_rate_limiter_exists(self):
        """Test that rate limiter is configured."""
        assert _plugins_limiter is not None
        assert _plugins_limiter.requests_per_minute == 60

    def test_rate_limit_exceeded(self, handler):
        """Test rate limit enforcement."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        # Exhaust rate limit
        for _ in range(61):
            _plugins_limiter.is_allowed("127.0.0.1")

        with patch(
            "aragora.server.handlers.features.plugins.get_client_ip",
            return_value="127.0.0.1",
        ):
            result = handler.handle("/api/v1/plugins/available", {}, mock_handler)
            assert result.status == 429
