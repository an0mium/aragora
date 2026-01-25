"""Plugin and laboratory endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

# Common parameter definitions for reuse
_PLUGIN_NAME_PARAM = {
    "name": "name",
    "in": "path",
    "required": True,
    "schema": {"type": "string"},
    "description": "Plugin name (lowercase alphanumeric with hyphens)",
}


def _plugin_list_endpoint(deprecated: bool = False) -> dict:
    """Generate plugin list endpoint definition."""
    result = {
        "get": {
            "tags": ["Plugins"],
            "summary": "List plugins",
            "responses": {"200": _ok_response("Plugin list")},
        },
    }
    if deprecated:
        result["get"]["deprecated"] = True
        result["get"]["description"] = (
            "Deprecated: Use /api/v1/plugins instead. Sunset: 2026-12-31."
        )
    return result


def _plugin_details_endpoint(deprecated: bool = False) -> dict:
    """Generate plugin details endpoint definition."""
    result = {
        "get": {
            "tags": ["Plugins"],
            "summary": "Get plugin details",
            "parameters": [_PLUGIN_NAME_PARAM],
            "responses": {"200": _ok_response("Plugin details")},
        },
    }
    if deprecated:
        result["get"]["deprecated"] = True
        result["get"]["description"] = (
            "Deprecated: Use /api/v1/plugins/{name} instead. Sunset: 2026-12-31."
        )
    return result


def _plugin_run_endpoint(deprecated: bool = False) -> dict:
    """Generate plugin run endpoint definition."""
    result = {
        "post": {
            "tags": ["Plugins"],
            "summary": "Run plugin",
            "parameters": [_PLUGIN_NAME_PARAM],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {"200": _ok_response("Plugin result")},
            "security": [{"bearerAuth": []}],
        },
    }
    if deprecated:
        result["post"]["deprecated"] = True
        result["post"]["description"] = (
            "Deprecated: Use /api/v1/plugins/{name}/run instead. Sunset: 2026-12-31."
        )
    return result


def _plugin_install_endpoint(deprecated: bool = False) -> dict:
    """Generate plugin install/uninstall endpoint definition."""
    result = {
        "post": {
            "tags": ["Plugins"],
            "summary": "Install plugin",
            "parameters": [_PLUGIN_NAME_PARAM],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {"200": _ok_response("Installation result")},
            "security": [{"bearerAuth": []}],
        },
        "delete": {
            "tags": ["Plugins"],
            "summary": "Uninstall plugin",
            "parameters": [_PLUGIN_NAME_PARAM],
            "responses": {"200": _ok_response("Uninstallation result")},
            "security": [{"bearerAuth": []}],
        },
    }
    if deprecated:
        result["post"]["deprecated"] = True
        result["post"]["description"] = (
            "Deprecated: Use /api/v1/plugins/{name}/install instead. Sunset: 2026-12-31."
        )
        result["delete"]["deprecated"] = True
        result["delete"]["description"] = (
            "Deprecated: Use /api/v1/plugins/{name}/install instead. Sunset: 2026-12-31."
        )
    return result


def _plugin_installed_endpoint(deprecated: bool = False) -> dict:
    """Generate installed plugins list endpoint definition."""
    result = {
        "get": {
            "tags": ["Plugins"],
            "summary": "List installed plugins",
            "responses": {"200": _ok_response("Installed plugins list")},
            "security": [{"bearerAuth": []}],
        },
    }
    if deprecated:
        result["get"]["deprecated"] = True
        result["get"]["description"] = (
            "Deprecated: Use /api/v1/plugins/installed instead. Sunset: 2026-12-31."
        )
    return result


def _plugin_marketplace_endpoint(deprecated: bool = False) -> dict:
    """Generate marketplace endpoint definition."""
    result = {
        "get": {
            "tags": ["Plugins"],
            "summary": "Get marketplace listings",
            "responses": {"200": _ok_response("Marketplace data with categories")},
        },
    }
    if deprecated:
        result["get"]["deprecated"] = True
        result["get"]["description"] = (
            "Deprecated: Use /api/v1/plugins/marketplace instead. Sunset: 2026-12-31."
        )
    return result


def _plugin_submit_endpoint(deprecated: bool = False) -> dict:
    """Generate plugin submission endpoint definition."""
    result = {
        "post": {
            "tags": ["Plugins"],
            "summary": "Submit plugin for review",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {"200": _ok_response("Submission confirmation")},
            "security": [{"bearerAuth": []}],
        },
    }
    if deprecated:
        result["post"]["deprecated"] = True
        result["post"]["description"] = (
            "Deprecated: Use /api/v1/plugins/submit instead. Sunset: 2026-12-31."
        )
    return result


def _plugin_submissions_endpoint(deprecated: bool = False) -> dict:
    """Generate submissions list endpoint definition."""
    result = {
        "get": {
            "tags": ["Plugins"],
            "summary": "List user's plugin submissions",
            "responses": {"200": _ok_response("Submissions list")},
            "security": [{"bearerAuth": []}],
        },
    }
    if deprecated:
        result["get"]["deprecated"] = True
        result["get"]["description"] = (
            "Deprecated: Use /api/v1/plugins/submissions instead. Sunset: 2026-12-31."
        )
    return result


PLUGIN_ENDPOINTS = {
    # Versioned endpoints (preferred)
    "/api/v1/plugins": _plugin_list_endpoint(),
    "/api/v1/plugins/installed": _plugin_installed_endpoint(),
    "/api/v1/plugins/marketplace": _plugin_marketplace_endpoint(),
    "/api/v1/plugins/submit": _plugin_submit_endpoint(),
    "/api/v1/plugins/submissions": _plugin_submissions_endpoint(),
    "/api/v1/plugins/{name}": _plugin_details_endpoint(),
    "/api/v1/plugins/{name}/run": _plugin_run_endpoint(),
    "/api/v1/plugins/{name}/install": _plugin_install_endpoint(),
    # Legacy endpoints (deprecated, sunset 2026-12-31)
    "/api/plugins": _plugin_list_endpoint(deprecated=True),
    "/api/plugins/installed": _plugin_installed_endpoint(deprecated=True),
    "/api/plugins/marketplace": _plugin_marketplace_endpoint(deprecated=True),
    "/api/plugins/submit": _plugin_submit_endpoint(deprecated=True),
    "/api/plugins/submissions": _plugin_submissions_endpoint(deprecated=True),
    "/api/plugins/{name}": _plugin_details_endpoint(deprecated=True),
    "/api/plugins/{name}/run": _plugin_run_endpoint(deprecated=True),
    "/api/plugins/{name}/install": _plugin_install_endpoint(deprecated=True),
    "/api/laboratory/emergent-traits": {
        "get": {
            "tags": ["Laboratory"],
            "summary": "Emergent traits",
            "parameters": [
                {
                    "name": "min_confidence",
                    "in": "query",
                    "schema": {"type": "number", "default": 0.5},
                },
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}},
            ],
            "responses": {"200": _ok_response("Emergent traits")},
        },
    },
    "/api/laboratory/cross-pollinations/suggest": {
        "get": {
            "tags": ["Laboratory"],
            "summary": "Cross-pollination suggestions",
            "responses": {"200": _ok_response("Suggestions")},
        },
    },
}
