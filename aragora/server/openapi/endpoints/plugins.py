"""Plugin and laboratory endpoint definitions."""

from typing import Any

from aragora.server.openapi.helpers import _ok_response

# Common parameter definitions for reuse
_PLUGIN_NAME_PARAM: dict[str, Any] = {
    "name": "name",
    "in": "path",
    "required": True,
    "schema": {"type": "string"},
    "description": "Plugin name (lowercase alphanumeric with hyphens)",
}


def _plugin_list_endpoint(deprecated: bool = False, versioned: bool = True) -> dict[str, Any]:
    """Generate plugin list endpoint definition."""
    op_id = "listPlugins" if versioned else "listPluginsLegacy"
    result: dict[str, Any] = {
        "get": {
            "tags": ["Plugins"],
            "summary": "List plugins",
            "operationId": op_id,
            "description": "Get list of all available plugins." if not deprecated else None,
            "responses": {"200": _ok_response("Plugin list")},
        },
    }
    if deprecated:
        result["get"]["deprecated"] = True
        result["get"]["description"] = (
            "Deprecated: Use /api/v1/plugins instead. Sunset: 2026-12-31."
        )
    return result


def _plugin_details_endpoint(deprecated: bool = False, versioned: bool = True) -> dict[str, Any]:
    """Generate plugin details endpoint definition."""
    op_id = "getPlugin" if versioned else "getPluginLegacy"
    result: dict[str, Any] = {
        "get": {
            "tags": ["Plugins"],
            "summary": "Get plugin details",
            "operationId": op_id,
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


def _plugin_run_endpoint(deprecated: bool = False, versioned: bool = True) -> dict[str, Any]:
    """Generate plugin run endpoint definition."""
    op_id = "runPlugin" if versioned else "runPluginLegacy"
    result: dict[str, Any] = {
        "post": {
            "tags": ["Plugins"],
            "summary": "Run plugin",
            "operationId": op_id,
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


def _plugin_install_endpoint(deprecated: bool = False, versioned: bool = True) -> dict[str, Any]:
    """Generate plugin install/uninstall endpoint definition."""
    install_id = "installPlugin" if versioned else "installPluginLegacy"
    uninstall_id = "uninstallPlugin" if versioned else "uninstallPluginLegacy"
    result: dict[str, Any] = {
        "post": {
            "tags": ["Plugins"],
            "summary": "Install plugin",
            "operationId": install_id,
            "parameters": [_PLUGIN_NAME_PARAM],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {"200": _ok_response("Installation result")},
            "security": [{"bearerAuth": []}],
        },
        "delete": {
            "tags": ["Plugins"],
            "summary": "Uninstall plugin",
            "operationId": uninstall_id,
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


def _plugin_installed_endpoint(deprecated: bool = False, versioned: bool = True) -> dict[str, Any]:
    """Generate installed plugins list endpoint definition."""
    op_id = "listInstalledPlugins" if versioned else "listInstalledPluginsLegacy"
    result: dict[str, Any] = {
        "get": {
            "tags": ["Plugins"],
            "summary": "List installed plugins",
            "operationId": op_id,
            "description": "Get list of plugins installed for the current user."
            if not deprecated
            else None,
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


def _plugin_marketplace_endpoint(
    deprecated: bool = False, versioned: bool = True
) -> dict[str, Any]:
    """Generate marketplace endpoint definition."""
    op_id = "getPluginMarketplace" if versioned else "getPluginMarketplaceLegacy"
    result: dict[str, Any] = {
        "get": {
            "tags": ["Plugins"],
            "summary": "Get marketplace listings",
            "operationId": op_id,
            "description": "Get plugin marketplace with categories and featured plugins."
            if not deprecated
            else None,
            "responses": {"200": _ok_response("Marketplace data with categories")},
        },
    }
    if deprecated:
        result["get"]["deprecated"] = True
        result["get"]["description"] = (
            "Deprecated: Use /api/v1/plugins/marketplace instead. Sunset: 2026-12-31."
        )
    return result


def _plugin_submit_endpoint(deprecated: bool = False, versioned: bool = True) -> dict[str, Any]:
    """Generate plugin submission endpoint definition."""
    op_id = "submitPlugin" if versioned else "submitPluginLegacy"
    result: dict[str, Any] = {
        "post": {
            "tags": ["Plugins"],
            "summary": "Submit plugin for review",
            "operationId": op_id,
            "description": "Submit a new plugin for marketplace review."
            if not deprecated
            else None,
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


def _plugin_submissions_endpoint(
    deprecated: bool = False, versioned: bool = True
) -> dict[str, Any]:
    """Generate submissions list endpoint definition."""
    op_id = "listPluginSubmissions" if versioned else "listPluginSubmissionsLegacy"
    result: dict[str, Any] = {
        "get": {
            "tags": ["Plugins"],
            "summary": "List user's plugin submissions",
            "operationId": op_id,
            "description": "Get list of plugins submitted by the current user for review."
            if not deprecated
            else None,
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
    "/api/v1/plugins": _plugin_list_endpoint(versioned=True),
    "/api/v1/plugins/installed": _plugin_installed_endpoint(versioned=True),
    "/api/v1/plugins/marketplace": _plugin_marketplace_endpoint(versioned=True),
    "/api/v1/plugins/submit": _plugin_submit_endpoint(versioned=True),
    "/api/v1/plugins/submissions": _plugin_submissions_endpoint(versioned=True),
    "/api/v1/plugins/{name}": _plugin_details_endpoint(versioned=True),
    "/api/v1/plugins/{name}/run": _plugin_run_endpoint(versioned=True),
    "/api/v1/plugins/{name}/install": _plugin_install_endpoint(versioned=True),
    # Legacy endpoints (deprecated, sunset 2026-12-31)
    "/api/plugins": _plugin_list_endpoint(deprecated=True, versioned=False),
    "/api/plugins/installed": _plugin_installed_endpoint(deprecated=True, versioned=False),
    "/api/plugins/marketplace": _plugin_marketplace_endpoint(deprecated=True, versioned=False),
    "/api/plugins/submit": _plugin_submit_endpoint(deprecated=True, versioned=False),
    "/api/plugins/submissions": _plugin_submissions_endpoint(deprecated=True, versioned=False),
    "/api/plugins/{name}": _plugin_details_endpoint(deprecated=True, versioned=False),
    "/api/plugins/{name}/run": _plugin_run_endpoint(deprecated=True, versioned=False),
    "/api/plugins/{name}/install": _plugin_install_endpoint(deprecated=True, versioned=False),
    "/api/laboratory/emergent-traits": {
        "get": {
            "tags": ["Laboratory"],
            "summary": "Emergent traits",
            "operationId": "listLaboratoryEmergentTraits",
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
            "operationId": "listLaboratoryCrossPollinationsSuggest",
            "responses": {"200": _ok_response("Suggestions")},
        },
    },
}
