"""Memory endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

MEMORY_ENDPOINTS = {
    "/api/v1/memory/continuum/retrieve": {
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
    "/api/v1/memory/continuum/consolidate": {
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
    "/api/v1/memory/continuum/cleanup": {
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
    "/api/v1/memory/tier-stats": {
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
    "/api/v1/memory/archive-stats": {
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
    "/api/v1/memory/pressure": {
        "get": {
            "tags": ["Memory"],
            "summary": "Memory pressure",
            "operationId": "getMemoryPressure",
            "description": "Get memory pressure and per-tier utilization.",
            "responses": {
                "200": _ok_response(
                    "Memory pressure",
                    {
                        "pressure": {"type": "number"},
                        "status": {"type": "string"},
                        "tier_utilization": {"type": "object"},
                        "total_memories": {"type": "integer"},
                        "cleanup_recommended": {"type": "boolean"},
                    },
                )
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/memory/tiers": {
        "get": {
            "tags": ["Memory"],
            "summary": "List memory tiers",
            "operationId": "listMemoryTiers",
            "description": "List all memory tiers with utilization stats.",
            "responses": {
                "200": _ok_response(
                    "Memory tiers",
                    {
                        "tiers": {"type": "array", "items": {"type": "object"}},
                        "total_memories": {"type": "integer"},
                        "transitions_24h": {"type": "integer"},
                    },
                )
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/memory/search": {
        "get": {
            "tags": ["Memory"],
            "summary": "Search memories",
            "operationId": "listMemorySearch",
            "description": "Search continuum memory across tiers.",
            "parameters": [
                {"name": "q", "in": "query", "schema": {"type": "string"}, "required": True},
                {
                    "name": "tier",
                    "in": "query",
                    "schema": {"type": "string"},
                    "description": "Comma-separated tier filter",
                },
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}},
                {
                    "name": "min_importance",
                    "in": "query",
                    "schema": {"type": "number", "default": 0.0},
                },
                {
                    "name": "sort",
                    "in": "query",
                    "schema": {
                        "type": "string",
                        "enum": ["relevance", "importance", "recency"],
                    },
                },
            ],
            "responses": {
                "200": _ok_response(
                    "Search results",
                    {
                        "query": {"type": "string"},
                        "results": {"type": "array", "items": {"type": "object"}},
                        "count": {"type": "integer"},
                        "tiers_searched": {"type": "array", "items": {"type": "string"}},
                    },
                )
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/memory/search-index": {
        "get": {
            "tags": ["Memory"],
            "summary": "Progressive search index",
            "operationId": "listMemorySearchIndex",
            "description": "Stage 1 search: compact index results with previews.",
            "parameters": [
                {"name": "q", "in": "query", "schema": {"type": "string"}, "required": True},
                {"name": "tier", "in": "query", "schema": {"type": "string"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}},
                {
                    "name": "min_importance",
                    "in": "query",
                    "schema": {"type": "number", "default": 0.0},
                },
                {"name": "use_hybrid", "in": "query", "schema": {"type": "boolean"}},
                {"name": "include_external", "in": "query", "schema": {"type": "boolean"}},
                {"name": "external", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response(
                    "Search index results",
                    {
                        "query": {"type": "string"},
                        "results": {"type": "array", "items": {"type": "object"}},
                        "count": {"type": "integer"},
                        "external_results": {"type": "array", "items": {"type": "object"}},
                    },
                )
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/memory/search-timeline": {
        "get": {
            "tags": ["Memory"],
            "summary": "Progressive search timeline",
            "operationId": "listMemorySearchTimeline",
            "description": "Stage 2 search: timeline around an anchor memory.",
            "parameters": [
                {
                    "name": "anchor_id",
                    "in": "query",
                    "schema": {"type": "string"},
                    "required": True,
                },
                {"name": "before", "in": "query", "schema": {"type": "integer", "default": 3}},
                {"name": "after", "in": "query", "schema": {"type": "integer", "default": 3}},
                {"name": "tier", "in": "query", "schema": {"type": "string"}},
                {
                    "name": "min_importance",
                    "in": "query",
                    "schema": {"type": "number", "default": 0.0},
                },
            ],
            "responses": {
                "200": _ok_response(
                    "Timeline results",
                    {
                        "anchor_id": {"type": "string"},
                        "anchor": {"type": "object"},
                        "before": {"type": "array", "items": {"type": "object"}},
                        "after": {"type": "array", "items": {"type": "object"}},
                    },
                )
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/memory/entries": {
        "get": {
            "tags": ["Memory"],
            "summary": "Fetch memory entries",
            "operationId": "listMemoryEntries",
            "description": "Stage 3 search: fetch full memory entries by ID.",
            "parameters": [
                {
                    "name": "ids",
                    "in": "query",
                    "schema": {"type": "string"},
                    "required": True,
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Entries",
                    {
                        "ids": {"type": "array", "items": {"type": "string"}},
                        "entries": {"type": "array", "items": {"type": "object"}},
                        "count": {"type": "integer"},
                    },
                )
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/memory/viewer": {
        "get": {
            "tags": ["Memory"],
            "summary": "Memory viewer UI",
            "operationId": "getMemoryViewer",
            "description": "HTML viewer for progressive memory search.",
            "responses": {
                "200": {
                    "description": "Memory viewer HTML",
                    "content": {"text/html": {"schema": {"type": "string"}}},
                }
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/memory/critiques": {
        "get": {
            "tags": ["Memory"],
            "summary": "List critiques",
            "operationId": "listMemoryCritiques",
            "description": "Browse critique store entries.",
            "responses": {
                "200": _ok_response(
                    "Critiques",
                    {
                        "critiques": {"type": "array", "items": {"type": "object"}},
                        "count": {"type": "integer"},
                        "total": {"type": "integer"},
                    },
                )
            },
            "security": [{"bearerAuth": []}],
        },
    },
}
