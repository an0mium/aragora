"""Tests for plugins handler.

Tests the plugins API endpoints including:
- GET /api/plugins - List all available plugins
- GET /api/plugins/{name} - Get details for a specific plugin
- POST /api/plugins/{name}/run - Run a plugin with provided input
- GET /api/plugins/installed - List installed plugins for user/org
- POST /api/plugins/{name}/install - Install a plugin
- DELETE /api/plugins/{name}/install - Uninstall a plugin
- POST /api/plugins/submit - Submit a new plugin for review
- GET /api/plugins/submissions - List user's plugin submissions
- GET /api/plugins/marketplace - Get marketplace listings

Both versioned (/api/v1/plugins/...) and legacy (/api/plugins/...) paths tested.
"""

import json
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.features.plugins import (
    PluginsHandler,
    _installed_plugins,
    _plugin_submissions,
    SUBMISSION_STATUS_PENDING,
)
from aragora.server.handlers.base import HandlerResult

_TEST_TOKEN = "test-token-abc123"


# =============================================================================
# Helpers
# =============================================================================


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _make_handler(body: dict | None = None, path: str | None = None) -> MagicMock:
    """Create a mock HTTP handler with optional JSON body and path.

    Includes an Authorization header with a test Bearer token so that
    the @require_auth decorator passes.
    """
    handler = MagicMock()
    if body is not None:
        body_bytes = json.dumps(body).encode()
        handler.rfile = BytesIO(body_bytes)
        handler.headers = {
            "Content-Length": str(len(body_bytes)),
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_TEST_TOKEN}",
        }
    else:
        handler.rfile = BytesIO(b"{}")
        handler.headers = {
            "Content-Length": "2",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_TEST_TOKEN}",
        }
    handler.client_address = ("127.0.0.1", 12345)
    if path is not None:
        handler.path = path
    else:
        # Remove path attr so _get_original_path returns None
        del handler.path
    return handler


class MockPlugin:
    """Mock plugin object returned by registry."""

    def __init__(self, name: str = "test-plugin", featured: bool = False, category: str = "other"):
        self.name = name
        self.featured = featured
        self.category = category

    def to_dict(self) -> dict:
        return {"name": self.name, "category": self.category}


class MockRunner:
    """Mock plugin runner."""

    def __init__(self, valid: bool = True, missing: list[str] | None = None):
        self._valid = valid
        self._missing = missing or []

    def _validate_requirements(self) -> tuple[bool, list[str]]:
        return self._valid, self._missing


class MockRunResult:
    """Mock result from running a plugin."""

    def to_dict(self) -> dict:
        return {"status": "success", "output": "result data"}


class MockRegistry:
    """Mock plugin registry."""

    def __init__(
        self,
        plugins: list | None = None,
        runners: dict[str, MockRunner] | None = None,
    ):
        self._plugins = plugins or []
        self._runners = runners or {}
        self._manifests: dict[str, MockPlugin] = {p.name: p for p in self._plugins}

    def list_plugins(self) -> list:
        return self._plugins

    def get(self, name: str):
        return self._manifests.get(name)

    def get_runner(self, name: str):
        return self._runners.get(name)

    async def run_plugin(self, name, input_data, config, working_dir):
        return MockRunResult()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_plugin_state():
    """Reset in-memory plugin stores before each test."""
    _installed_plugins.clear()
    _plugin_submissions.clear()
    yield
    _installed_plugins.clear()
    _plugin_submissions.clear()


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Reset rate limiters to avoid cross-test pollution."""
    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters
        reset_rate_limiters()
    except ImportError:
        pass
    yield
    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters
        reset_rate_limiters()
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def _bypass_require_auth(monkeypatch):
    """Patch auth_config so @require_auth decorator passes with our test token."""
    from aragora.server import auth as _auth_mod

    monkeypatch.setattr(_auth_mod.auth_config, "api_token", _TEST_TOKEN)
    monkeypatch.setattr(_auth_mod.auth_config, "validate_token", lambda token, **kw: token == _TEST_TOKEN)


@pytest.fixture
def handler():
    """Create a PluginsHandler instance."""
    return PluginsHandler(server_context={})


@pytest.fixture
def mock_registry():
    """Create a mock registry with a sample plugin."""
    plugin = MockPlugin(name="test-plugin", featured=False, category="utility")
    runner = MockRunner(valid=True)
    return MockRegistry(plugins=[plugin], runners={"test-plugin": runner})


# =============================================================================
# can_handle Tests
# =============================================================================


class TestCanHandle:
    """Tests for can_handle path routing."""

    def test_exact_versioned_paths(self, handler):
        assert handler.can_handle("/api/v1/plugins") is True
        assert handler.can_handle("/api/v1/plugins/installed") is True
        assert handler.can_handle("/api/v1/plugins/marketplace") is True
        assert handler.can_handle("/api/v1/plugins/submit") is True
        assert handler.can_handle("/api/v1/plugins/submissions") is True

    def test_exact_legacy_paths(self, handler):
        assert handler.can_handle("/api/plugins") is True
        assert handler.can_handle("/api/plugins/installed") is True
        assert handler.can_handle("/api/plugins/marketplace") is True
        assert handler.can_handle("/api/plugins/submit") is True
        assert handler.can_handle("/api/plugins/submissions") is True

    def test_versioned_plugin_name(self, handler):
        assert handler.can_handle("/api/v1/plugins/my-plugin") is True

    def test_versioned_plugin_run(self, handler):
        assert handler.can_handle("/api/v1/plugins/my-plugin/run") is True

    def test_versioned_plugin_install(self, handler):
        assert handler.can_handle("/api/v1/plugins/my-plugin/install") is True

    def test_legacy_plugin_name(self, handler):
        assert handler.can_handle("/api/plugins/my-plugin") is True

    def test_legacy_plugin_run(self, handler):
        assert handler.can_handle("/api/plugins/my-plugin/run") is True

    def test_legacy_plugin_install(self, handler):
        assert handler.can_handle("/api/plugins/my-plugin/install") is True

    def test_unrelated_paths_rejected(self, handler):
        assert handler.can_handle("/api/debates") is False
        assert handler.can_handle("/api/v1/debates") is False
        assert handler.can_handle("/api/users") is False
        assert handler.can_handle("/other") is False

    def test_deeply_nested_versioned_rejected(self, handler):
        """Too many segments should be rejected."""
        assert handler.can_handle("/api/v1/plugins/a/b/c") is False

    def test_deeply_nested_legacy_rejected(self, handler):
        assert handler.can_handle("/api/plugins/a/b/c") is False


# =============================================================================
# Initialization Tests
# =============================================================================


class TestPluginsHandlerInit:
    """Tests for handler initialization."""

    def test_init_with_server_context(self):
        h = PluginsHandler(server_context={"key": "val"})
        assert h.ctx == {"key": "val"}

    def test_init_with_ctx(self):
        h = PluginsHandler(ctx={"k": "v"})
        assert h.ctx == {"k": "v"}

    def test_init_defaults_to_empty(self):
        h = PluginsHandler()
        assert h.ctx == {}

    def test_routes_defined(self, handler):
        assert hasattr(handler, "ROUTES")
        assert len(handler.ROUTES) > 0

    def test_sunset_date_set(self, handler):
        assert handler._SUNSET_DATE is not None
        assert "2026" in handler._SUNSET_DATE


# =============================================================================
# GET /api/plugins - List Plugins
# =============================================================================


class TestListPlugins:
    """Tests for listing all available plugins."""

    def test_list_plugins_unavailable(self, handler):
        """Returns 503 when plugins module not available."""
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", False):
            result = handler.handle("/api/v1/plugins", {}, _make_handler())
        assert _status(result) == 503

    def test_list_plugins_registry_none(self, handler):
        """Returns 503 when get_registry is None."""
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", None):
            result = handler.handle("/api/v1/plugins", {}, _make_handler())
        assert _status(result) == 503

    def test_list_plugins_success(self, handler, mock_registry):
        """Returns plugin list on success."""
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=mock_registry):
            result = handler.handle("/api/v1/plugins", {}, _make_handler())
        body = _body(result)
        assert _status(result) == 200
        assert body["count"] == 1
        assert body["plugins"][0]["name"] == "test-plugin"

    def test_list_plugins_empty_registry(self, handler):
        registry = MockRegistry(plugins=[])
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=registry):
            result = handler.handle("/api/v1/plugins", {}, _make_handler())
        body = _body(result)
        assert body["count"] == 0
        assert body["plugins"] == []

    def test_list_plugins_legacy_path(self, handler, mock_registry):
        """Legacy path works and adds Sunset header."""
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=mock_registry):
            result = handler.handle("/api/plugins", {}, _make_handler())
        assert _status(result) == 200
        assert result.headers.get("Sunset") is not None
        assert result.headers.get("Deprecation") == "true"

    def test_list_plugins_versioned_no_sunset(self, handler, mock_registry):
        """Versioned path does NOT add Sunset header."""
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=mock_registry):
            result = handler.handle("/api/v1/plugins", {}, _make_handler())
        assert result.headers.get("Sunset") is None


# =============================================================================
# GET /api/plugins/{name} - Get Plugin Details
# =============================================================================


class TestGetPlugin:
    """Tests for getting a specific plugin's details."""

    def test_get_plugin_success_with_runner(self, handler, mock_registry):
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=mock_registry):
            result = handler.handle("/api/v1/plugins/test-plugin", {}, _make_handler())
        body = _body(result)
        assert _status(result) == 200
        assert body["name"] == "test-plugin"
        assert body["requirements_satisfied"] is True
        assert body["missing_requirements"] == []

    def test_get_plugin_success_no_runner(self, handler):
        plugin = MockPlugin(name="simple-plugin")
        registry = MockRegistry(plugins=[plugin], runners={})
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=registry):
            result = handler.handle("/api/v1/plugins/simple-plugin", {}, _make_handler())
        body = _body(result)
        assert _status(result) == 200
        assert body["name"] == "simple-plugin"
        assert "requirements_satisfied" not in body

    def test_get_plugin_not_found(self, handler):
        registry = MockRegistry(plugins=[])
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=registry):
            result = handler.handle("/api/v1/plugins/nonexistent", {}, _make_handler())
        assert _status(result) == 404

    def test_get_plugin_unavailable(self, handler):
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", False):
            result = handler.handle("/api/v1/plugins/test-plugin", {}, _make_handler())
        assert _status(result) == 503

    def test_get_plugin_invalid_name(self, handler):
        """Plugin name with invalid characters returns 400."""
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=MockRegistry()):
            result = handler.handle("/api/v1/plugins/../../etc", {}, _make_handler())
        assert _status(result) == 400

    def test_get_plugin_legacy_path(self, handler, mock_registry):
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=mock_registry):
            result = handler.handle("/api/plugins/test-plugin", {}, _make_handler())
        assert _status(result) == 200
        assert result.headers.get("Sunset") is not None

    def test_get_plugin_with_missing_requirements(self, handler):
        plugin = MockPlugin(name="complex-plugin")
        runner = MockRunner(valid=False, missing=["numpy", "pandas"])
        registry = MockRegistry(plugins=[plugin], runners={"complex-plugin": runner})
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=registry):
            result = handler.handle("/api/v1/plugins/complex-plugin", {}, _make_handler())
        body = _body(result)
        assert body["requirements_satisfied"] is False
        assert "numpy" in body["missing_requirements"]


# =============================================================================
# GET /api/plugins/marketplace - Marketplace
# =============================================================================


class TestGetMarketplace:
    """Tests for the marketplace endpoint."""

    def test_marketplace_unavailable(self, handler):
        """Returns empty marketplace when plugins not available."""
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", False):
            result = handler.handle("/api/v1/plugins/marketplace", {}, _make_handler())
        body = _body(result)
        assert _status(result) == 200
        assert body["total"] == 0
        assert body["featured"] == []
        assert body["message"] == "Plugin system not configured"

    def test_marketplace_with_plugins(self, handler):
        plugins = [
            MockPlugin(name="alpha", featured=True, category="analytics"),
            MockPlugin(name="beta", featured=False, category="utility"),
            MockPlugin(name="gamma", featured=True, category="analytics"),
        ]
        registry = MockRegistry(plugins=plugins)
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=registry):
            result = handler.handle("/api/v1/plugins/marketplace", {}, _make_handler())
        body = _body(result)
        assert body["total"] == 3
        assert len(body["featured"]) == 2
        assert "analytics" in body["categories"]
        assert "utility" in body["categories"]

    def test_marketplace_featured_limited_to_5(self, handler):
        """Featured list should be capped at 5."""
        plugins = [MockPlugin(name=f"p{i}", featured=True, category="cat") for i in range(8)]
        registry = MockRegistry(plugins=plugins)
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=registry):
            result = handler.handle("/api/v1/plugins/marketplace", {}, _make_handler())
        body = _body(result)
        assert len(body["featured"]) == 5

    def test_marketplace_legacy_path(self, handler, mock_registry):
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=mock_registry):
            result = handler.handle("/api/plugins/marketplace", {}, _make_handler())
        assert _status(result) == 200
        assert result.headers.get("Sunset") is not None


# =============================================================================
# GET /api/plugins/installed - List Installed
# =============================================================================


class TestListInstalled:
    """Tests for listing installed plugins."""

    def test_list_installed_empty(self, handler):
        """No installed plugins returns empty list."""
        result = handler.handle("/api/v1/plugins/installed", {}, _make_handler())
        body = _body(result)
        assert _status(result) == 200
        assert body["count"] == 0
        assert body["installed"] == []

    def test_list_installed_with_plugins(self, handler, mock_registry):
        """Returns enriched installed plugins."""
        _installed_plugins["test-user-001"] = {
            "test-plugin": {"installed_at": "2026-01-01T00:00:00", "config": {"key": "val"}}
        }
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=mock_registry):
            result = handler.handle("/api/v1/plugins/installed", {}, _make_handler())
        body = _body(result)
        assert body["count"] == 1
        assert body["installed"][0]["name"] == "test-plugin"
        assert body["installed"][0]["installed_at"] == "2026-01-01T00:00:00"
        assert body["installed"][0]["user_config"] == {"key": "val"}

    def test_list_installed_missing_manifest(self, handler):
        """Installed plugins with missing manifest are excluded."""
        _installed_plugins["test-user-001"] = {
            "gone-plugin": {"installed_at": "2026-01-01T00:00:00", "config": {}}
        }
        registry = MockRegistry(plugins=[])  # Empty -- manifest not found
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=registry):
            result = handler.handle("/api/v1/plugins/installed", {}, _make_handler())
        body = _body(result)
        assert body["count"] == 0

    def test_list_installed_no_plugins_module(self, handler):
        """Returns empty when plugins module not available."""
        _installed_plugins["test-user-001"] = {
            "test-plugin": {"installed_at": "2026-01-01T00:00:00", "config": {}}
        }
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", False):
            result = handler.handle("/api/v1/plugins/installed", {}, _make_handler())
        body = _body(result)
        assert body["count"] == 0

    def test_list_installed_legacy_path(self, handler):
        result = handler.handle("/api/plugins/installed", {}, _make_handler())
        assert _status(result) == 200
        assert result.headers.get("Sunset") is not None


# =============================================================================
# GET /api/plugins/submissions - List Submissions
# =============================================================================


class TestListSubmissions:
    """Tests for listing user's plugin submissions."""

    def test_list_submissions_empty(self, handler):
        result = handler.handle("/api/v1/plugins/submissions", {}, _make_handler())
        body = _body(result)
        assert _status(result) == 200
        assert body["count"] == 0
        assert body["submissions"] == []

    def test_list_submissions_with_entries(self, handler):
        _plugin_submissions["sub-001"] = {
            "id": "sub-001",
            "submitted_by": "test-user-001",
            "submitted_at": "2026-02-01T00:00:00",
            "status": "pending",
            "manifest": {"name": "my-plugin"},
        }
        _plugin_submissions["sub-002"] = {
            "id": "sub-002",
            "submitted_by": "test-user-001",
            "submitted_at": "2026-02-02T00:00:00",
            "status": "approved",
            "manifest": {"name": "other-plugin"},
        }
        result = handler.handle("/api/v1/plugins/submissions", {}, _make_handler())
        body = _body(result)
        assert body["count"] == 2
        # Should be sorted newest first
        assert body["submissions"][0]["id"] == "sub-002"

    def test_list_submissions_filters_by_user(self, handler):
        """Only returns submissions for the current user."""
        _plugin_submissions["sub-001"] = {
            "id": "sub-001",
            "submitted_by": "test-user-001",
            "submitted_at": "2026-01-01T00:00:00",
            "status": "pending",
        }
        _plugin_submissions["sub-002"] = {
            "id": "sub-002",
            "submitted_by": "other-user",
            "submitted_at": "2026-01-02T00:00:00",
            "status": "pending",
        }
        result = handler.handle("/api/v1/plugins/submissions", {}, _make_handler())
        body = _body(result)
        assert body["count"] == 1
        assert body["submissions"][0]["id"] == "sub-001"

    def test_list_submissions_legacy_path(self, handler):
        result = handler.handle("/api/plugins/submissions", {}, _make_handler())
        assert _status(result) == 200
        assert result.headers.get("Sunset") is not None


# =============================================================================
# POST /api/plugins/{name}/install - Install Plugin
# =============================================================================


class TestInstallPlugin:
    """Tests for installing a plugin."""

    def test_install_success(self, handler, mock_registry):
        body_handler = _make_handler(body={"config": {"setting": "on"}})
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=mock_registry):
            result = handler.handle_post("/api/v1/plugins/test-plugin/install", {}, body_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["success"] is True
        assert "test-plugin" in _installed_plugins.get("test-user-001", {})

    def test_install_already_installed(self, handler, mock_registry):
        _installed_plugins["test-user-001"] = {
            "test-plugin": {"installed_at": "2026-01-01T00:00:00", "config": {}}
        }
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=mock_registry):
            result = handler.handle_post("/api/v1/plugins/test-plugin/install", {}, _make_handler())
        body = _body(result)
        assert body["already_installed"] is True

    def test_install_plugin_not_found(self, handler):
        registry = MockRegistry(plugins=[])
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=registry):
            result = handler.handle_post("/api/v1/plugins/nonexistent/install", {}, _make_handler())
        assert _status(result) == 404

    def test_install_missing_requirements(self, handler):
        plugin = MockPlugin(name="needs-deps")
        runner = MockRunner(valid=False, missing=["numpy"])
        registry = MockRegistry(plugins=[plugin], runners={"needs-deps": runner})
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=registry):
            result = handler.handle_post("/api/v1/plugins/needs-deps/install", {}, _make_handler())
        assert _status(result) == 400
        assert "numpy" in _body(result).get("error", "")

    def test_install_plugins_unavailable(self, handler):
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", False):
            result = handler.handle_post("/api/v1/plugins/test-plugin/install", {}, _make_handler())
        assert _status(result) == 503

    def test_install_no_config_in_body(self, handler, mock_registry):
        """Installing without config uses empty dict as default."""
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=mock_registry):
            result = handler.handle_post("/api/v1/plugins/test-plugin/install", {}, _make_handler())
        body = _body(result)
        assert body["success"] is True
        install_info = _installed_plugins["test-user-001"]["test-plugin"]
        assert install_info["config"] == {}

    def test_install_legacy_path(self, handler, mock_registry):
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=mock_registry):
            result = handler.handle_post("/api/plugins/test-plugin/install", {}, _make_handler())
        assert _status(result) == 200
        assert result.headers.get("Sunset") is not None


# =============================================================================
# DELETE /api/plugins/{name}/install - Uninstall Plugin
# =============================================================================


class TestUninstallPlugin:
    """Tests for uninstalling a plugin."""

    def test_uninstall_success(self, handler):
        _installed_plugins["test-user-001"] = {
            "test-plugin": {"installed_at": "2026-01-01T00:00:00", "config": {}}
        }
        result = handler.handle_delete("/api/v1/plugins/test-plugin/install", {}, _make_handler())
        body = _body(result)
        assert _status(result) == 200
        assert body["success"] is True
        assert "test-plugin" not in _installed_plugins.get("test-user-001", {})

    def test_uninstall_not_installed(self, handler):
        result = handler.handle_delete("/api/v1/plugins/test-plugin/install", {}, _make_handler())
        assert _status(result) == 404

    def test_uninstall_legacy_path(self, handler):
        _installed_plugins["test-user-001"] = {
            "test-plugin": {"installed_at": "2026-01-01T00:00:00", "config": {}}
        }
        result = handler.handle_delete("/api/plugins/test-plugin/install", {}, _make_handler())
        assert _status(result) == 200
        assert result.headers.get("Sunset") is not None


# =============================================================================
# POST /api/plugins/{name}/run - Run Plugin
# =============================================================================


class TestRunPlugin:
    """Tests for running a plugin."""

    def test_run_success(self, handler, mock_registry):
        body_data = {"input": {}, "config": {}, "working_dir": "."}
        body_handler = _make_handler(body=body_data)
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=mock_registry), \
             patch("aragora.server.handlers.features.plugins.validate_against_schema") as mock_validate, \
             patch("aragora.server.handlers.features.plugins.run_async", return_value=MockRunResult()):
            mock_validate.return_value = MagicMock(is_valid=True)
            result = handler.handle_post("/api/v1/plugins/test-plugin/run", {}, body_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["status"] == "success"

    def test_run_plugins_unavailable(self, handler):
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", False):
            result = handler.handle_post("/api/v1/plugins/test-plugin/run", {}, _make_handler(body={}))
        assert _status(result) == 503

    def test_run_invalid_body(self, handler):
        """Invalid/missing JSON body returns 400."""
        bad_handler = MagicMock()
        bad_handler.rfile = BytesIO(b"not json")
        bad_handler.headers = {
            "Content-Length": "8",
            "Authorization": f"Bearer {_TEST_TOKEN}",
        }
        bad_handler.client_address = ("127.0.0.1", 12345)
        del bad_handler.path
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=MockRegistry()):
            result = handler.handle_post("/api/v1/plugins/test-plugin/run", {}, bad_handler)
        assert _status(result) == 400

    def test_run_schema_validation_failure(self, handler, mock_registry):
        body_handler = _make_handler(body={"input": {}})
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=mock_registry), \
             patch("aragora.server.handlers.features.plugins.validate_against_schema") as mock_validate:
            mock_validate.return_value = MagicMock(is_valid=False, error="Schema invalid")
            result = handler.handle_post("/api/v1/plugins/test-plugin/run", {}, body_handler)
        assert _status(result) == 400

    def test_run_plugin_not_found(self, handler):
        registry = MockRegistry(plugins=[])
        body_handler = _make_handler(body={"input": {}})
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=registry), \
             patch("aragora.server.handlers.features.plugins.validate_against_schema") as mock_validate:
            mock_validate.return_value = MagicMock(is_valid=True)
            result = handler.handle_post("/api/v1/plugins/nonexistent/run", {}, body_handler)
        assert _status(result) == 404

    def test_run_working_dir_traversal(self, handler, mock_registry):
        """Working dir outside cwd should be rejected."""
        body_handler = _make_handler(body={"input": {}, "working_dir": "/etc"})
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=mock_registry), \
             patch("aragora.server.handlers.features.plugins.validate_against_schema") as mock_validate:
            mock_validate.return_value = MagicMock(is_valid=True)
            result = handler.handle_post("/api/v1/plugins/test-plugin/run", {}, body_handler)
        assert _status(result) == 400
        assert "Working directory" in _body(result).get("error", "")

    def test_run_legacy_path(self, handler, mock_registry):
        body_handler = _make_handler(body={"input": {}})
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=mock_registry), \
             patch("aragora.server.handlers.features.plugins.validate_against_schema") as mock_validate, \
             patch("aragora.server.handlers.features.plugins.run_async", return_value=MockRunResult()):
            mock_validate.return_value = MagicMock(is_valid=True)
            result = handler.handle_post("/api/plugins/test-plugin/run", {}, body_handler)
        assert _status(result) == 200
        assert result.headers.get("Sunset") is not None


# =============================================================================
# POST /api/plugins/submit - Submit Plugin
# =============================================================================


class TestSubmitPlugin:
    """Tests for submitting a plugin for review."""

    def _make_submit_body(self, name: str = "new-plugin", **overrides) -> dict:
        manifest = {
            "name": name,
            "version": "1.0.0",
            "description": "A plugin",
            "entry_point": "my_plugin.main:run",  # module.path:function format
        }
        body = {"manifest": manifest, "source_url": "https://github.com/example/repo", "notes": "Please review"}
        body.update(overrides)
        return body

    def test_submit_success(self, handler):
        body_data = self._make_submit_body()
        body_handler = _make_handler(body=body_data)
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", False), \
             patch("aragora.server.handlers.features.plugins.validate_against_schema") as mock_validate:
            mock_validate.return_value = MagicMock(is_valid=True)
            result = handler.handle_post("/api/v1/plugins/submit", {}, body_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["success"] is True
        assert body["status"] == "pending"
        assert "submission_id" in body

    def test_submit_missing_manifest(self, handler):
        body_handler = _make_handler(body={"source_url": "https://example.com"})
        with patch("aragora.server.handlers.features.plugins.validate_against_schema") as mock_validate:
            mock_validate.return_value = MagicMock(is_valid=True)
            result = handler.handle_post("/api/v1/plugins/submit", {}, body_handler)
        assert _status(result) == 400
        assert "manifest" in _body(result).get("error", "").lower()

    def test_submit_invalid_body(self, handler):
        bad_handler = MagicMock()
        bad_handler.rfile = BytesIO(b"not json")
        bad_handler.headers = {
            "Content-Length": "8",
            "Authorization": f"Bearer {_TEST_TOKEN}",
        }
        bad_handler.client_address = ("127.0.0.1", 12345)
        del bad_handler.path
        result = handler.handle_post("/api/v1/plugins/submit", {}, bad_handler)
        assert _status(result) == 400

    def test_submit_manifest_schema_validation_fails(self, handler):
        body_data = self._make_submit_body()
        body_handler = _make_handler(body=body_data)
        # _submit_plugin does a local import from aragora.server.validation.schema
        with patch("aragora.server.validation.schema.validate_against_schema") as mock_validate:
            mock_validate.return_value = MagicMock(is_valid=False, error="Bad manifest field")
            result = handler.handle_post("/api/v1/plugins/submit", {}, body_handler)
        assert _status(result) == 400

    def test_submit_duplicate_name_in_marketplace(self, handler, mock_registry):
        """Rejects submission when plugin name already exists in marketplace."""
        body_data = self._make_submit_body(name="test-plugin")
        body_handler = _make_handler(body=body_data)
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=mock_registry), \
             patch("aragora.server.handlers.features.plugins.validate_against_schema") as mock_validate:
            mock_validate.return_value = MagicMock(is_valid=True)
            result = handler.handle_post("/api/v1/plugins/submit", {}, body_handler)
        assert _status(result) == 409
        assert "already exists" in _body(result).get("error", "")

    def test_submit_duplicate_pending_submission(self, handler):
        """Rejects when user already has a pending submission with same name."""
        _plugin_submissions["sub-001"] = {
            "id": "sub-001",
            "submitted_by": "test-user-001",
            "submitted_at": "2026-01-01T00:00:00",
            "status": SUBMISSION_STATUS_PENDING,
            "manifest": {"name": "new-plugin"},
        }
        body_data = self._make_submit_body(name="new-plugin")
        body_handler = _make_handler(body=body_data)
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", False), \
             patch("aragora.server.handlers.features.plugins.validate_against_schema") as mock_validate:
            mock_validate.return_value = MagicMock(is_valid=True)
            result = handler.handle_post("/api/v1/plugins/submit", {}, body_handler)
        assert _status(result) == 409
        assert "pending" in _body(result).get("error", "").lower()

    def test_submit_same_name_different_user_ok(self, handler):
        """Different user can submit same name."""
        _plugin_submissions["sub-001"] = {
            "id": "sub-001",
            "submitted_by": "other-user",
            "submitted_at": "2026-01-01T00:00:00",
            "status": SUBMISSION_STATUS_PENDING,
            "manifest": {"name": "new-plugin"},
        }
        body_data = self._make_submit_body(name="new-plugin")
        body_handler = _make_handler(body=body_data)
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", False), \
             patch("aragora.server.handlers.features.plugins.validate_against_schema") as mock_validate:
            mock_validate.return_value = MagicMock(is_valid=True)
            result = handler.handle_post("/api/v1/plugins/submit", {}, body_handler)
        assert _status(result) == 200

    def test_submit_approved_same_name_ok(self, handler):
        """A previously approved submission with same name doesn't block new one."""
        _plugin_submissions["sub-001"] = {
            "id": "sub-001",
            "submitted_by": "test-user-001",
            "submitted_at": "2026-01-01T00:00:00",
            "status": "approved",
            "manifest": {"name": "new-plugin"},
        }
        body_data = self._make_submit_body(name="new-plugin")
        body_handler = _make_handler(body=body_data)
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", False), \
             patch("aragora.server.handlers.features.plugins.validate_against_schema") as mock_validate:
            mock_validate.return_value = MagicMock(is_valid=True)
            result = handler.handle_post("/api/v1/plugins/submit", {}, body_handler)
        assert _status(result) == 200

    def test_submit_with_plugin_manifest_validation(self, handler):
        """When PLUGINS_AVAILABLE, validates using PluginManifest."""
        body_data = self._make_submit_body()
        body_handler = _make_handler(body=body_data)

        mock_manifest_cls = MagicMock()
        mock_temp = MagicMock()
        mock_temp.validate.return_value = (True, [])
        mock_manifest_cls.from_dict.return_value = mock_temp

        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=MockRegistry()), \
             patch("aragora.server.handlers.features.plugins.validate_against_schema") as mock_validate, \
             patch("aragora.plugins.manifest.PluginManifest", mock_manifest_cls):
            mock_validate.return_value = MagicMock(is_valid=True)
            result = handler.handle_post("/api/v1/plugins/submit", {}, body_handler)
        assert _status(result) == 200

    def test_submit_manifest_object_validation_fails(self, handler):
        """When PluginManifest.validate() returns errors, reject."""
        body_data = self._make_submit_body()
        body_handler = _make_handler(body=body_data)

        mock_manifest_cls = MagicMock()
        mock_temp = MagicMock()
        mock_temp.validate.return_value = (False, ["missing entry_point"])
        mock_manifest_cls.from_dict.return_value = mock_temp

        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=MockRegistry()), \
             patch("aragora.server.handlers.features.plugins.validate_against_schema") as mock_validate, \
             patch("aragora.plugins.manifest.PluginManifest", mock_manifest_cls):
            mock_validate.return_value = MagicMock(is_valid=True)
            result = handler.handle_post("/api/v1/plugins/submit", {}, body_handler)
        assert _status(result) == 400
        assert "validation failed" in _body(result).get("error", "").lower()

    def test_submit_manifest_from_dict_exception(self, handler):
        """PluginManifest.from_dict() raising ValueError is handled."""
        body_data = self._make_submit_body()
        body_handler = _make_handler(body=body_data)

        mock_manifest_cls = MagicMock()
        mock_manifest_cls.from_dict.side_effect = ValueError("bad format")

        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=MockRegistry()), \
             patch("aragora.server.handlers.features.plugins.validate_against_schema") as mock_validate, \
             patch("aragora.plugins.manifest.PluginManifest", mock_manifest_cls):
            mock_validate.return_value = MagicMock(is_valid=True)
            result = handler.handle_post("/api/v1/plugins/submit", {}, body_handler)
        assert _status(result) == 400
        assert "invalid manifest" in _body(result).get("error", "").lower()

    def test_submit_legacy_path(self, handler):
        body_data = self._make_submit_body()
        body_handler = _make_handler(body=body_data)
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", False), \
             patch("aragora.server.handlers.features.plugins.validate_against_schema") as mock_validate:
            mock_validate.return_value = MagicMock(is_valid=True)
            result = handler.handle_post("/api/plugins/submit", {}, body_handler)
        assert _status(result) == 200
        assert result.headers.get("Sunset") is not None

    def test_submit_stores_submission(self, handler):
        """Verify submission is stored in _plugin_submissions."""
        body_data = self._make_submit_body()
        body_handler = _make_handler(body=body_data)
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", False), \
             patch("aragora.server.handlers.features.plugins.validate_against_schema") as mock_validate:
            mock_validate.return_value = MagicMock(is_valid=True)
            result = handler.handle_post("/api/v1/plugins/submit", {}, body_handler)
        body = _body(result)
        sub_id = body["submission_id"]
        assert sub_id in _plugin_submissions
        stored = _plugin_submissions[sub_id]
        assert stored["submitted_by"] == "test-user-001"
        assert stored["status"] == SUBMISSION_STATUS_PENDING
        assert stored["manifest"]["name"] == "new-plugin"


# =============================================================================
# Sunset / Legacy Header Tests
# =============================================================================


class TestSunsetHeaders:
    """Tests for RFC 8594 Sunset header behavior."""

    def test_legacy_path_detected(self, handler):
        assert handler._is_legacy_path("/api/plugins") is True
        assert handler._is_legacy_path("/api/plugins/installed") is True

    def test_versioned_path_not_legacy(self, handler):
        assert handler._is_legacy_path("/api/v1/plugins") is False
        assert handler._is_legacy_path("/api/v1/plugins/my-plugin") is False

    def test_sunset_header_added_for_legacy(self, handler, mock_registry):
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=mock_registry):
            result = handler.handle("/api/plugins", {}, _make_handler())
        assert result.headers["Sunset"] == handler._SUNSET_DATE
        assert result.headers["Deprecation"] == "true"

    def test_no_sunset_header_for_versioned(self, handler, mock_registry):
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=mock_registry):
            result = handler.handle("/api/v1/plugins", {}, _make_handler())
        assert result.headers.get("Sunset") is None

    def test_original_path_used_for_detection(self, handler, mock_registry):
        """When handler has path attr with /api/v1/, it's not treated as legacy."""
        h = _make_handler(path="/api/v1/plugins")
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=mock_registry):
            result = handler.handle("/api/plugins", {}, h)
        # handler.path says /api/v1/, so it should NOT be legacy
        assert result.headers.get("Sunset") is None

    def test_original_path_with_query_string(self, handler, mock_registry):
        """Query strings are stripped from original path for detection."""
        h = _make_handler(path="/api/plugins?foo=bar")
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=mock_registry):
            result = handler.handle("/api/plugins", {}, h)
        assert result.headers.get("Sunset") is not None


# =============================================================================
# Path Normalization Tests
# =============================================================================


class TestPathNormalization:
    """Tests for path normalization helpers."""

    def test_normalize_versioned_to_legacy(self, handler):
        assert handler._normalize_plugin_path("/api/v1/plugins/foo") == "/api/plugins/foo"

    def test_normalize_legacy_unchanged(self, handler):
        assert handler._normalize_plugin_path("/api/plugins/foo") == "/api/plugins/foo"

    def test_plugin_name_index_always_3(self, handler):
        assert handler._get_plugin_name_index("/api/plugins/foo") == 3

    def test_get_original_path_none_handler(self, handler):
        assert handler._get_original_path(None) is None

    def test_get_original_path_no_path_attr(self, handler):
        h = MagicMock(spec=[])
        assert handler._get_original_path(h) is None

    def test_get_original_path_non_string_path(self, handler):
        h = MagicMock()
        h.path = 12345
        assert handler._get_original_path(h) is None


# =============================================================================
# Route Dispatch (handle returns None for unmatched)
# =============================================================================


class TestRouteDispatch:
    """Tests for route dispatch returning None for unmatched paths."""

    def test_handle_returns_none_for_unmatched(self, handler):
        result = handler.handle("/api/v1/something-else", {}, _make_handler())
        assert result is None

    def test_handle_post_returns_none_for_unmatched(self, handler):
        result = handler.handle_post("/api/v1/something-else", {}, _make_handler())
        assert result is None

    def test_handle_delete_returns_none_for_unmatched(self, handler):
        result = handler.handle_delete("/api/v1/something-else", {}, _make_handler())
        assert result is None

    def test_handle_does_not_match_run_suffix(self, handler):
        """GET to /run path should not match GET handler."""
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=MockRegistry()):
            result = handler.handle("/api/v1/plugins/test/run", {}, _make_handler())
        # /run and /install are excluded from GET plugin details
        assert result is None

    def test_handle_does_not_match_install_suffix(self, handler):
        """GET to /install path should not match GET handler."""
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=MockRegistry()):
            result = handler.handle("/api/v1/plugins/test/install", {}, _make_handler())
        assert result is None


# =============================================================================
# get_user_id Tests
# =============================================================================


class TestGetUserId:
    """Tests for user ID extraction."""

    def test_get_user_id_success(self, handler):
        """User ID is extracted from authenticated context."""
        h = _make_handler()
        uid = handler.get_user_id(h)
        assert uid == "test-user-001"

    def test_get_user_id_no_user_id_attr(self, handler):
        """Returns None when user has no user_id attribute."""
        with patch.object(handler, "get_current_user") as mock_user:
            mock_obj = MagicMock(spec=[])  # Object with no user_id
            mock_user.return_value = mock_obj
            uid = handler.get_user_id(_make_handler())
        assert uid is None

    def test_get_user_id_no_user(self, handler):
        """Returns None when not authenticated."""
        with patch.object(handler, "get_current_user", return_value=None):
            uid = handler.get_user_id(_make_handler())
        assert uid is None


# =============================================================================
# Multiple Plugins in Registry
# =============================================================================


class TestMultiplePlugins:
    """Tests with multiple plugins in registry."""

    def test_list_multiple_plugins(self, handler):
        plugins = [
            MockPlugin(name="alpha"),
            MockPlugin(name="beta"),
            MockPlugin(name="gamma"),
        ]
        registry = MockRegistry(plugins=plugins)
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=registry):
            result = handler.handle("/api/v1/plugins", {}, _make_handler())
        body = _body(result)
        assert body["count"] == 3
        names = [p["name"] for p in body["plugins"]]
        assert "alpha" in names
        assert "beta" in names
        assert "gamma" in names

    def test_marketplace_grouping(self, handler):
        """Plugins are correctly grouped by category."""
        plugins = [
            MockPlugin(name="a", category="analytics"),
            MockPlugin(name="b", category="analytics"),
            MockPlugin(name="c", category="security"),
        ]
        registry = MockRegistry(plugins=plugins)
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=registry):
            result = handler.handle("/api/v1/plugins/marketplace", {}, _make_handler())
        body = _body(result)
        assert len(body["categories"]["analytics"]) == 2
        assert len(body["categories"]["security"]) == 1

    def test_install_multiple_plugins(self, handler):
        """User can install multiple plugins."""
        plugins = [MockPlugin(name="alpha"), MockPlugin(name="beta")]
        registry = MockRegistry(
            plugins=plugins,
            runners={"alpha": MockRunner(), "beta": MockRunner()},
        )
        with patch("aragora.server.handlers.features.plugins.PLUGINS_AVAILABLE", True), \
             patch("aragora.server.handlers.features.plugins.get_registry", return_value=registry):
            handler.handle_post("/api/v1/plugins/alpha/install", {}, _make_handler())
            handler.handle_post("/api/v1/plugins/beta/install", {}, _make_handler())
        user_plugins = _installed_plugins.get("test-user-001", {})
        assert "alpha" in user_plugins
        assert "beta" in user_plugins

    def test_uninstall_one_keeps_others(self, handler):
        """Uninstalling one plugin keeps others intact."""
        _installed_plugins["test-user-001"] = {
            "alpha": {"installed_at": "2026-01-01T00:00:00", "config": {}},
            "beta": {"installed_at": "2026-01-02T00:00:00", "config": {}},
        }
        handler.handle_delete("/api/v1/plugins/alpha/install", {}, _make_handler())
        assert "alpha" not in _installed_plugins["test-user-001"]
        assert "beta" in _installed_plugins["test-user-001"]
