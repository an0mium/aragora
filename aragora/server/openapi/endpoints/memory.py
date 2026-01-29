"""Memory endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

MEMORY_ENDPOINTS = {
    "/api/memory/continuum/retrieve": {
        "get": {
            "tags": ["Memory"],
            "summary": "Retrieve memories",
            "operationId": "listMemoryContinuumRetrieve",
            "description": """Retrieve memories from the continuum store.

The continuum memory system stores memories across four tiers:
- **fast**: Short-term memory (1 minute TTL)
- **medium**: Session memory (1 hour TTL)
- **slow**: Cross-session memory (1 day TTL)
- **glacial**: Long-term patterns (1 week TTL)

Memories are retrieved by semantic similarity to the query.""",
            "parameters": [
                {
                    "name": "query",
                    "in": "query",
                    "schema": {"type": "string"},
                    "description": "Semantic search query",
                    "required": True,
                },
                {
                    "name": "tier",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["fast", "medium", "slow", "glacial"]},
                    "description": "Filter by specific tier (optional)",
                },
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100},
                    "description": "Maximum memories to return",
                },
            ],
            "responses": {"200": _ok_response("Retrieved memories", "MemoryRetrievalResponse")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/memory/continuum/consolidate": {
        "post": {
            "tags": ["Memory"],
            "summary": "Consolidate memories",
            "operationId": "createMemoryContinuumConsolidate",
            "description": """Trigger memory consolidation across tiers.

Consolidation promotes frequently-accessed memories to faster tiers
and demotes stale memories to slower tiers based on access patterns.""",
            "responses": {
                "200": _ok_response("Consolidation completed", "MemoryConsolidationResult")
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/memory/continuum/cleanup": {
        "post": {
            "tags": ["Memory"],
            "summary": "Cleanup expired memories",
            "operationId": "createMemoryContinuumCleanup",
            "description": """Remove expired memories from all tiers.

Each tier has different TTLs. This operation removes memories
that have exceeded their tier's TTL threshold.""",
            "responses": {"200": _ok_response("Cleanup completed", "MemoryCleanupResult")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/memory/tier-stats": {
        "get": {
            "tags": ["Memory"],
            "summary": "Memory tier statistics",
            "operationId": "listMemoryTierStats",
            "description": """Get statistics for each memory tier.

Returns count, size, and age metrics for fast, medium, slow, and glacial tiers.""",
            "responses": {"200": _ok_response("Tier statistics", "MemoryTierStatsResponse")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/memory/archive-stats": {
        "get": {
            "tags": ["Memory"],
            "summary": "Archive statistics",
            "operationId": "listMemoryArchiveStats",
            "description": """Get statistics on archived memories and storage usage.

Archives contain memories that have been moved to cold storage for long-term retention.""",
            "responses": {"200": _ok_response("Archive statistics", "MemoryArchiveStats")},
            "security": [{"bearerAuth": []}],
        },
    },
}
