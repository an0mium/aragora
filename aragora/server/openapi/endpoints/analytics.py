"""Analytics and insights endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

ANALYTICS_ENDPOINTS = {
    "/api/analytics/disagreements": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Disagreement analysis",
            "description": "Get metrics on agent disagreement patterns",
            "responses": {"200": _ok_response("Disagreement statistics")},
        },
    },
    "/api/analytics/role-rotation": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Role rotation stats",
            "responses": {"200": _ok_response("Role rotation data")},
        },
    },
    "/api/analytics/early-stops": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Early stop statistics",
            "responses": {"200": _ok_response("Early stop data")},
        },
    },
    "/api/ranking/stats": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Ranking statistics",
            "responses": {"200": _ok_response("Ranking stats")},
        },
    },
    "/api/memory/stats": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Memory statistics",
            "responses": {"200": _ok_response("Memory stats")},
        },
    },
    "/api/flips/recent": {
        "get": {
            "tags": ["Insights"],
            "summary": "Recent position flips",
            "parameters": [
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}}
            ],
            "responses": {"200": _ok_response("Recent flips")},
        },
    },
    "/api/flips/summary": {
        "get": {
            "tags": ["Insights"],
            "summary": "Flip summary",
            "responses": {"200": _ok_response("Flip summary statistics")},
        },
    },
    "/api/insights/recent": {
        "get": {
            "tags": ["Insights"],
            "summary": "Recent insights",
            "parameters": [
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}}
            ],
            "responses": {"200": _ok_response("Recent insights")},
        },
    },
    "/api/insights/extract-detailed": {
        "post": {
            "tags": ["Insights"],
            "summary": "Extract detailed insights",
            "description": "Computationally expensive insight extraction (requires auth)",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {"200": _ok_response("Detailed insights")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/moments/summary": {
        "get": {
            "tags": ["Insights"],
            "summary": "Moments summary",
            "responses": {"200": _ok_response("Moments summary")},
        },
    },
    "/api/moments/timeline": {
        "get": {
            "tags": ["Insights"],
            "summary": "Moments timeline",
            "responses": {"200": _ok_response("Timeline data")},
        },
    },
    "/api/moments/trending": {
        "get": {
            "tags": ["Insights"],
            "summary": "Trending moments",
            "responses": {"200": _ok_response("Trending moments")},
        },
    },
    "/api/moments/by-type/{type}": {
        "get": {
            "tags": ["Insights"],
            "summary": "Moments by type",
            "parameters": [
                {"name": "type", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Moments of specified type")},
        },
    },
}
