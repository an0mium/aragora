"""Memory endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

MEMORY_ENDPOINTS = {
    "/api/memory/continuum/retrieve": {
        "get": {
            "tags": ["Memory"],
            "summary": "Retrieve memories",
            "operationId": "listMemoryContinuumRetrieve",
            "description": "Retrieve memories from continuum store",
            "parameters": [
                {"name": "query", "in": "query", "schema": {"type": "string"}},
                {
                    "name": "tier",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["fast", "medium", "slow", "glacial"]},
                },
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 10}},
            ],
            "responses": {"200": _ok_response("Retrieved memories")},
        },
    },
    "/api/memory/continuum/consolidate": {
        "post": {
            "tags": ["Memory"],
            "summary": "Consolidate memories",
            "operationId": "createMemoryContinuumConsolidate",
            "description": "Trigger memory consolidation across tiers",
            "responses": {"200": _ok_response("Consolidation result")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/memory/continuum/cleanup": {
        "post": {
            "tags": ["Memory"],
            "summary": "Cleanup memories",
            "operationId": "createMemoryContinuumCleanup",
            "description": "Remove expired memories",
            "responses": {"200": _ok_response("Cleanup result")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/memory/tier-stats": {
        "get": {
            "tags": ["Memory"],
            "summary": "Memory tier statistics",
            "description": "Get statistics for each memory tier (fast, medium, slow, glacial).",
            "operationId": "listMemoryTierStats",
            "responses": {"200": _ok_response("Tier stats")},
        },
    },
    "/api/memory/archive-stats": {
        "get": {
            "tags": ["Memory"],
            "summary": "Archive statistics",
            "description": "Get statistics on archived memories and storage usage.",
            "operationId": "listMemoryArchiveStats",
            "responses": {"200": _ok_response("Archive stats")},
        },
    },
}
