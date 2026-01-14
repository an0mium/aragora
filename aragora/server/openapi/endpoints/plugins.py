"""Plugin and laboratory endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

PLUGIN_ENDPOINTS = {
    "/api/plugins": {
        "get": {
            "tags": ["Plugins"],
            "summary": "List plugins",
            "responses": {"200": _ok_response("Plugin list")},
        },
    },
    "/api/plugins/{name}": {
        "get": {
            "tags": ["Plugins"],
            "summary": "Get plugin details",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Plugin details")},
        },
    },
    "/api/plugins/{name}/run": {
        "post": {
            "tags": ["Plugins"],
            "summary": "Run plugin",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {"200": _ok_response("Plugin result")},
            "security": [{"bearerAuth": []}],
        },
    },
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
