"""Analytics and insights endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

ANALYTICS_ENDPOINTS = {
    "/api/analytics/disagreements": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Disagreement analysis",
            "operationId": "listAnalyticsDisagreements",
            "description": "Get metrics on agent disagreement patterns across debates.",
            "responses": {"200": _ok_response("Disagreement statistics")},
        },
    },
    "/api/analytics/role-rotation": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Role rotation stats",
            "operationId": "listAnalyticsRoleRotation",
            "description": "Get statistics on how agents rotate between proposer, critic, and judge roles.",
            "responses": {"200": _ok_response("Role rotation data")},
        },
    },
    "/api/analytics/early-stops": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Early stop statistics",
            "operationId": "listAnalyticsEarlyStops",
            "description": "Get data on debates that ended early due to early consensus or other conditions.",
            "responses": {"200": _ok_response("Early stop data")},
        },
    },
    "/api/ranking/stats": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Ranking statistics",
            "operationId": "listRankingStats",
            "description": "Get aggregate ELO ranking statistics across all agents.",
            "responses": {"200": _ok_response("Ranking stats")},
        },
    },
    "/api/memory/stats": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Memory statistics",
            "operationId": "listMemoryStats",
            "description": "Get statistics on memory system usage and performance.",
            "responses": {"200": _ok_response("Memory stats")},
        },
    },
    "/api/flips/recent": {
        "get": {
            "tags": ["Insights"],
            "summary": "Recent position flips",
            "operationId": "listFlipsRecent",
            "description": "Get recent instances where agents changed their positions during debate.",
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
            "operationId": "listFlipsSummary",
            "description": "Get summary statistics on position flips and conviction changes.",
            "responses": {"200": _ok_response("Flip summary statistics")},
        },
    },
    "/api/insights/recent": {
        "get": {
            "tags": ["Insights"],
            "summary": "Recent insights",
            "operationId": "listInsightsRecent",
            "description": "Get recent insights extracted from debates.",
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
            "operationId": "createInsightsExtractDetailed",
            "description": "Computationally expensive insight extraction (requires auth).",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {"200": _ok_response("Detailed insights")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/moments/summary": {
        "get": {
            "tags": ["Insights"],
            "summary": "Moments summary",
            "operationId": "listMomentsSummary",
            "description": "Get summary of key moments across debates.",
            "responses": {"200": _ok_response("Moments summary")},
        },
    },
    "/api/moments/timeline": {
        "get": {
            "tags": ["Insights"],
            "summary": "Moments timeline",
            "operationId": "listMomentsTimeline",
            "description": "Get chronological timeline of significant debate moments.",
            "responses": {"200": _ok_response("Timeline data")},
        },
    },
    "/api/moments/trending": {
        "get": {
            "tags": ["Insights"],
            "summary": "Trending moments",
            "operationId": "listMomentsTrending",
            "description": "Get currently trending debate moments based on engagement.",
            "responses": {"200": _ok_response("Trending moments")},
        },
    },
    "/api/moments/by-type/{type}": {
        "get": {
            "tags": ["Insights"],
            "summary": "Moments by type",
            "operationId": "getMomentsByType",
            "description": "Get debate moments filtered by type (e.g., breakthrough, conflict, consensus).",
            "parameters": [
                {"name": "type", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Moments of specified type")},
        },
    },
}
