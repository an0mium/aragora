"""
Plugins endpoint handlers.

Endpoints:
- GET /api/plugins - List all available plugins
- GET /api/plugins/{name} - Get details for a specific plugin
- POST /api/plugins/{name}/run - Run a plugin with provided input
- GET /api/plugins/installed - List installed plugins for user/org
- POST /api/plugins/{name}/install - Install a plugin
- DELETE /api/plugins/{name}/install - Uninstall a plugin
- POST /api/plugins/submit - Submit a new plugin for review
- GET /api/plugins/submissions - List user's plugin submissions
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from aragora.server.http_utils import run_async
from aragora.server.middleware.rate_limit import rate_limit
from aragora.server.validation.schema import PLUGIN_RUN_SCHEMA, validate_against_schema
from aragora.utils.optional_imports import try_import

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    require_auth,
)

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_plugin_imports, PLUGINS_AVAILABLE = try_import("aragora.plugins.runner", "get_registry")
get_registry = _plugin_imports.get("get_registry")

# In-memory store for installed plugins per user/org
# Structure: {user_id: {plugin_name: {"installed_at": timestamp, "config": {...}}}}
_installed_plugins: dict[str, dict[str, dict]] = {}

# In-memory store for plugin submissions (pending review)
# Structure: {submission_id: {manifest, status, submitted_by, submitted_at, ...}}
_plugin_submissions: dict[str, dict] = {}

# Submission statuses
SUBMISSION_STATUS_PENDING = "pending"
SUBMISSION_STATUS_APPROVED = "approved"
SUBMISSION_STATUS_REJECTED = "rejected"


class PluginsHandler(BaseHandler):
    """Handler for plugins endpoints."""

    def get_user_id(self, handler) -> Optional[str]:
        """Extract user ID from authenticated request.

        Args:
            handler: HTTP request handler with headers

        Returns:
            User ID string if authenticated, None otherwise
        """
        user = self.get_current_user(handler)
        if user is None:
            return None
        return getattr(user, "user_id", None)

    ROUTES = [
        "/api/v1/plugins",
        "/api/v1/plugins/installed",
        "/api/v1/plugins/marketplace",
        "/api/v1/plugins/submit",
        "/api/v1/plugins/submissions",
        "/api/v1/plugins/*",
        "/api/v1/plugins/*/install",
        "/api/v1/plugins/*/run",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in (
            "/api/v1/plugins",
            "/api/v1/plugins/installed",
            "/api/v1/plugins/marketplace",
            "/api/v1/plugins/submit",
            "/api/v1/plugins/submissions",
        ):
            return True
        # Match /api/v1/plugins/{name}, /api/v1/plugins/{name}/run, /api/v1/plugins/{name}/install
        if path.startswith("/api/v1/plugins/"):
            parts = path.split("/")
            # /api/v1/plugins/{name} has 5 parts: ['', 'api', 'v1', 'plugins', '{name}']
            # /api/v1/plugins/{name}/run or /install has 6 parts
            return len(parts) in (5, 6)
        return False

    def handle(self, path: str, query_params: dict, handler=None) -> Optional[HandlerResult]:
        """Route GET requests to appropriate methods."""
        if path == "/api/v1/plugins":
            return self._list_plugins()

        if path == "/api/v1/plugins/installed":
            return self._list_installed(handler)

        if path == "/api/v1/plugins/marketplace":
            return self._get_marketplace()

        if path == "/api/v1/plugins/submissions":
            return self._list_submissions(handler)

        # Get plugin details: /api/plugins/{name}
        if path.startswith("/api/v1/plugins/") and not path.endswith(("/run", "/install")):
            plugin_name, err = self.extract_path_param(path, 3, "plugin_name")
            if err:
                return err
            return self._get_plugin(plugin_name)

        return None

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route POST requests to appropriate methods."""
        # Submit plugin: /api/plugins/submit
        if path == "/api/v1/plugins/submit":
            return self._submit_plugin(handler)

        # Run plugin: /api/plugins/{name}/run
        if path.startswith("/api/v1/plugins/") and path.endswith("/run"):
            plugin_name, err = self.extract_path_param(path, 3, "plugin_name")
            if err:
                return err
            return self._run_plugin(plugin_name, handler)

        # Install plugin: /api/plugins/{name}/install
        if path.startswith("/api/v1/plugins/") and path.endswith("/install"):
            plugin_name, err = self.extract_path_param(path, 3, "plugin_name")
            if err:
                return err
            return self._install_plugin(plugin_name, handler)

        return None

    def handle_delete(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route DELETE requests to appropriate methods."""
        # Uninstall plugin: /api/plugins/{name}/install
        if path.startswith("/api/v1/plugins/") and path.endswith("/install"):
            plugin_name, err = self.extract_path_param(path, 3, "plugin_name")
            if err:
                return err
            return self._uninstall_plugin(plugin_name, handler)
        return None

    @handle_errors("list plugins")
    def _list_plugins(self) -> HandlerResult:
        """List all available plugins."""
        if not PLUGINS_AVAILABLE or not get_registry:
            return error_response("Plugins module not available", 503)

        registry = get_registry()
        plugins = registry.list_plugins()
        return json_response(
            {
                "plugins": [p.to_dict() for p in plugins],
                "count": len(plugins),
            }
        )

    @handle_errors("get marketplace")
    def _get_marketplace(self) -> HandlerResult:
        """Get marketplace listings with categorized plugins.

        Returns featured plugins and plugins organized by category.
        """
        if not PLUGINS_AVAILABLE or not get_registry:
            return json_response(
                {
                    "featured": [],
                    "categories": {},
                    "total": 0,
                    "message": "Plugin system not configured",
                }
            )

        registry = get_registry()
        plugins = registry.list_plugins()

        # Categorize plugins
        featured = []
        categories: dict[str, list] = {}

        for plugin in plugins:
            plugin_dict = plugin.to_dict()

            # Check for featured flag
            if getattr(plugin, "featured", False):
                featured.append(plugin_dict)

            # Group by category
            category = getattr(plugin, "category", "other")
            if category not in categories:
                categories[category] = []
            categories[category].append(plugin_dict)

        return json_response(
            {
                "featured": featured[:5],  # Top 5 featured
                "categories": categories,
                "total": len(plugins),
            }
        )

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
            return json_response(
                {
                    **manifest.to_dict(),
                    "requirements_satisfied": valid,
                    "missing_requirements": missing,
                }
            )
        else:
            return json_response(manifest.to_dict())

    @require_auth
    @rate_limit(requests_per_minute=20, burst=5, limiter_name="plugin_run")
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

        # Schema validation for input sanitization
        validation_result = validate_against_schema(body, PLUGIN_RUN_SCHEMA)
        if not validation_result.is_valid:
            return error_response(validation_result.error, 400)

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
        result = run_async(registry.run_plugin(plugin_name, input_data, config, working_dir))
        return json_response(result.to_dict())

    @require_auth
    @handle_errors("list installed plugins")
    def _list_installed(self, handler) -> HandlerResult:
        """List installed plugins for the authenticated user.

        Returns:
            List of installed plugins with installation metadata.
        """
        user_id = self.get_user_id(handler)
        if not user_id:
            return error_response("Authentication required", 401)

        user_plugins = _installed_plugins.get(user_id, {})

        # Enrich with plugin details
        plugins_list = []
        if PLUGINS_AVAILABLE and get_registry:
            registry = get_registry()
            for name, install_info in user_plugins.items():
                manifest = registry.get(name)
                if manifest:
                    plugins_list.append(
                        {
                            **manifest.to_dict(),
                            "installed_at": install_info.get("installed_at"),
                            "user_config": install_info.get("config", {}),
                        }
                    )

        return json_response(
            {
                "installed": plugins_list,
                "count": len(plugins_list),
            }
        )

    @require_auth
    @rate_limit(requests_per_minute=30, burst=10, limiter_name="plugin_install")
    @handle_errors("install plugin")
    def _install_plugin(self, plugin_name: str, handler) -> HandlerResult:
        """Install a plugin for the authenticated user.

        POST body (optional):
            config: Initial configuration for the plugin

        Returns:
            Installation confirmation.
        """
        if not PLUGINS_AVAILABLE or not get_registry:
            return error_response("Plugins module not available", 503)

        user_id = self.get_user_id(handler)
        if not user_id:
            return error_response("Authentication required", 401)

        registry = get_registry()
        manifest = registry.get(plugin_name)
        if not manifest:
            return error_response(f"Plugin not found: {plugin_name}", 404)

        # Check requirements
        runner = registry.get_runner(plugin_name)
        if runner:
            valid, missing = runner._validate_requirements()
            if not valid:
                return error_response(f"Missing requirements: {', '.join(missing)}", 400)

        # Read optional config from body
        body = self.read_json_body(handler) or {}
        config = body.get("config", {})

        # Initialize user's plugins if needed
        if user_id not in _installed_plugins:
            _installed_plugins[user_id] = {}

        # Check if already installed
        if plugin_name in _installed_plugins[user_id]:
            return json_response(
                {
                    "success": True,
                    "message": f"Plugin {plugin_name} already installed",
                    "plugin": manifest.to_dict(),
                    "already_installed": True,
                }
            )

        # Install the plugin
        _installed_plugins[user_id][plugin_name] = {
            "installed_at": datetime.now().isoformat(),
            "config": config,
        }

        return json_response(
            {
                "success": True,
                "message": f"Plugin {plugin_name} installed successfully",
                "plugin": manifest.to_dict(),
                "installed_at": _installed_plugins[user_id][plugin_name]["installed_at"],
            }
        )

    @require_auth
    @handle_errors("uninstall plugin")
    def _uninstall_plugin(self, plugin_name: str, handler) -> HandlerResult:
        """Uninstall a plugin for the authenticated user.

        Returns:
            Uninstallation confirmation.
        """
        user_id = self.get_user_id(handler)
        if not user_id:
            return error_response("Authentication required", 401)

        user_plugins = _installed_plugins.get(user_id, {})

        if plugin_name not in user_plugins:
            return error_response(f"Plugin {plugin_name} not installed", 404)

        # Remove the plugin
        del _installed_plugins[user_id][plugin_name]

        return json_response(
            {
                "success": True,
                "message": f"Plugin {plugin_name} uninstalled successfully",
            }
        )

    @require_auth
    @rate_limit(requests_per_minute=10, burst=3, limiter_name="plugin_submit")
    @handle_errors("submit plugin")
    def _submit_plugin(self, handler) -> HandlerResult:
        """Submit a new plugin for marketplace review.

        POST body:
            manifest: Plugin manifest (name, version, description, entry_point, etc.)
            source_url: URL to plugin source repository (GitHub, GitLab, etc.)
            notes: Optional notes for reviewers

        Returns:
            Submission confirmation with submission ID.
        """
        import uuid

        user_id = self.get_user_id(handler)
        if not user_id:
            return error_response("Authentication required", 401)

        # Read request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        manifest = body.get("manifest")
        if not manifest:
            return error_response("Missing 'manifest' field", 400)

        # Validate required manifest fields
        required_fields = ["name", "version", "description", "entry_point"]
        missing = [f for f in required_fields if not manifest.get(f)]
        if missing:
            return error_response(f"Missing required manifest fields: {', '.join(missing)}", 400)

        # Validate name format (alphanumeric with hyphens)
        import re

        name = manifest.get("name", "")
        if not re.match(r"^[a-z][a-z0-9-]*$", name):
            return error_response(
                "Plugin name must start with lowercase letter and contain only lowercase letters, numbers, and hyphens",
                400,
            )

        # Check for duplicate name
        if PLUGINS_AVAILABLE and get_registry:
            registry = get_registry()
            if registry.get(name):
                return error_response(f"Plugin '{name}' already exists in marketplace", 409)

        # Check if user already has a pending submission with this name
        for sub_id, sub in _plugin_submissions.items():
            if (
                sub.get("submitted_by") == user_id
                and sub.get("manifest", {}).get("name") == name
                and sub.get("status") == SUBMISSION_STATUS_PENDING
            ):
                return error_response(
                    f"You already have a pending submission for '{name}' (ID: {sub_id})", 409
                )

        # Create submission
        submission_id = str(uuid.uuid4())[:8]
        submission = {
            "id": submission_id,
            "manifest": manifest,
            "source_url": body.get("source_url"),
            "notes": body.get("notes"),
            "submitted_by": user_id,
            "submitted_at": datetime.now().isoformat(),
            "status": SUBMISSION_STATUS_PENDING,
            "review_notes": None,
            "reviewed_at": None,
            "reviewed_by": None,
        }
        _plugin_submissions[submission_id] = submission

        logger.info(f"Plugin submission received: {name} v{manifest.get('version')} by {user_id}")

        return json_response(
            {
                "success": True,
                "submission_id": submission_id,
                "message": f"Plugin '{name}' submitted for review",
                "status": SUBMISSION_STATUS_PENDING,
            }
        )

    @require_auth
    @handle_errors("list submissions")
    def _list_submissions(self, handler) -> HandlerResult:
        """List plugin submissions for the authenticated user.

        Returns:
            List of user's plugin submissions with status.
        """
        user_id = self.get_user_id(handler)
        if not user_id:
            return error_response("Authentication required", 401)

        user_submissions = [
            sub for sub in _plugin_submissions.values() if sub.get("submitted_by") == user_id
        ]

        # Sort by submission date (newest first)
        user_submissions.sort(key=lambda x: x.get("submitted_at", ""), reverse=True)

        return json_response(
            {
                "submissions": user_submissions,
                "count": len(user_submissions),
            }
        )
