"""
Plugins endpoint handlers.

Endpoints:
- GET /api/plugins - List all available plugins
- GET /api/plugins/{name} - Get details for a specific plugin
- POST /api/plugins/{name}/run - Run a plugin with provided input
"""

import logging
from pathlib import Path
from typing import Optional

from aragora.utils.optional_imports import try_import
from aragora.server.http_utils import run_async
from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    handle_errors,
)

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_plugin_imports, PLUGINS_AVAILABLE = try_import(
    "aragora.plugins.runner",
    "get_registry"
)
get_registry = _plugin_imports.get("get_registry")


class PluginsHandler(BaseHandler):
    """Handler for plugins endpoints."""

    ROUTES = [
        "/api/plugins",
        "/api/plugins/*",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path == "/api/plugins":
            return True
        # Match /api/plugins/{name} or /api/plugins/{name}/run
        if path.startswith("/api/plugins/"):
            parts = path.split("/")
            # /api/plugins/{name} has 4 parts, /api/plugins/{name}/run has 5
            return len(parts) in (4, 5)
        return False

    def handle(self, path: str, query_params: dict, handler=None) -> Optional[HandlerResult]:
        """Route GET requests to appropriate methods."""
        if path == "/api/plugins":
            return self._list_plugins()

        # Get plugin details: /api/plugins/{name}
        if path.startswith("/api/plugins/") and not path.endswith("/run"):
            plugin_name, err = self.extract_path_param(path, 2, "plugin_name")
            if err:
                return err
            return self._get_plugin(plugin_name)

        return None

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route POST requests to appropriate methods."""
        # Run plugin: /api/plugins/{name}/run
        if path.startswith("/api/plugins/") and path.endswith("/run"):
            plugin_name, err = self.extract_path_param(path, 2, "plugin_name")
            if err:
                return err
            return self._run_plugin(plugin_name, handler)
        return None

    @handle_errors("list plugins")
    def _list_plugins(self) -> HandlerResult:
        """List all available plugins."""
        if not PLUGINS_AVAILABLE or not get_registry:
            return error_response("Plugins module not available", 503)

        registry = get_registry()
        plugins = registry.list_plugins()
        return json_response({
            "plugins": [p.to_dict() for p in plugins],
            "count": len(plugins),
        })

    @handle_errors("get plugin")
    def _get_plugin(self, plugin_name: str) -> HandlerResult:
        """Get details for a specific plugin."""
        if not PLUGINS_AVAILABLE or not get_registry:
            return error_response("Plugins module not available", 503)

        registry = get_registry()
        manifest = registry.get(plugin_name)
        if not manifest:
            return error_response(f"Plugin not found: {plugin_name}", 404)

        # Also check if requirements are satisfied
        runner = registry.get_runner(plugin_name)
        if runner:
            valid, missing = runner._validate_requirements()
            return json_response({
                **manifest.to_dict(),
                "requirements_satisfied": valid,
                "missing_requirements": missing,
            })
        else:
            return json_response(manifest.to_dict())

    @handle_errors("run plugin")
    def _run_plugin(self, plugin_name: str, handler) -> HandlerResult:
        """Run a plugin with provided input.

        POST body:
            input: Input data for the plugin (default: {})
            config: Plugin configuration (default: {})
            working_dir: Working directory for execution (default: ".")

        Returns:
            Plugin execution result.
        """
        if not PLUGINS_AVAILABLE or not get_registry:
            return error_response("Plugins module not available", 503)

        # Read request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body or body too large", 400)

        input_data = body.get("input", {})
        config = body.get("config", {})
        working_dir = body.get("working_dir", ".")

        # Validate working_dir (must be under current directory for security)
        cwd = Path.cwd().resolve()
        work_path = Path(working_dir).resolve()
        if not str(work_path).startswith(str(cwd)):
            return error_response("Working directory must be under current directory", 400)

        registry = get_registry()
        manifest = registry.get(plugin_name)
        if not manifest:
            return error_response(f"Plugin not found: {plugin_name}", 404)

        # Run plugin with timeout
        # Use run_async() for safe sync/async bridging
        result = run_async(
            registry.run_plugin(plugin_name, input_data, config, working_dir)
        )
        return json_response(result.to_dict())
