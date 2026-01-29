"""
Memory and Knowledge OpenAPI Schema Definitions.

Schemas for memory tiers, knowledge mound, and related operations.
"""

from typing import Any

MEMORY_SCHEMAS: dict[str, Any] = {
    # Memory schemas
    "MemoryStats": {
        "type": "object",
        "description": "Memory system statistics",
        "properties": {
            "total_entries": {"type": "integer"},
            "by_tier": {
                "type": "object",
                "properties": {
                    "fast": {"type": "integer"},
                    "medium": {"type": "integer"},
                    "slow": {"type": "integer"},
                    "glacial": {"type": "integer"},
                },
            },
            "cache_hit_rate": {"type": "number"},
        },
    },
    "MemoryEntry": {
        "type": "object",
        "description": "A single memory entry from the continuum store",
        "properties": {
            "id": {"type": "string", "description": "Unique memory ID"},
            "content": {"type": "string", "description": "Memory content"},
            "tier": {
                "type": "string",
                "enum": ["fast", "medium", "slow", "glacial"],
                "description": "Memory tier",
            },
            "created_at": {"type": "string", "format": "date-time"},
            "expires_at": {"type": "string", "format": "date-time"},
            "relevance_score": {"type": "number", "description": "Relevance to query 0-1"},
            "metadata": {"type": "object", "additionalProperties": True},
        },
        "required": ["id", "content", "tier"],
    },
    "MemoryRetrievalResponse": {
        "type": "object",
        "description": "Response from memory retrieval",
        "properties": {
            "memories": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/MemoryEntry"},
            },
            "total": {"type": "integer"},
            "tier": {"type": "string"},
            "query": {"type": "string"},
        },
    },
    "MemoryTierStats": {
        "type": "object",
        "description": "Statistics for a single memory tier",
        "properties": {
            "tier": {"type": "string", "enum": ["fast", "medium", "slow", "glacial"]},
            "count": {"type": "integer"},
            "size_bytes": {"type": "integer"},
            "oldest_entry": {"type": "string", "format": "date-time"},
            "newest_entry": {"type": "string", "format": "date-time"},
            "avg_age_seconds": {"type": "number"},
        },
    },
    "MemoryTierStatsResponse": {
        "type": "object",
        "description": "Statistics for all memory tiers",
        "properties": {
            "tiers": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/MemoryTierStats"},
            },
            "total_memories": {"type": "integer"},
            "total_size_bytes": {"type": "integer"},
        },
    },
    "MemoryArchiveStats": {
        "type": "object",
        "description": "Archive statistics",
        "properties": {
            "archived_count": {"type": "integer"},
            "archive_size_bytes": {"type": "integer"},
            "oldest_archive": {"type": "string", "format": "date-time"},
            "compression_ratio": {"type": "number"},
        },
    },
    "MemoryConsolidationResult": {
        "type": "object",
        "description": "Result of memory consolidation",
        "properties": {
            "memories_processed": {"type": "integer"},
            "memories_promoted": {"type": "integer"},
            "memories_demoted": {"type": "integer"},
            "duration_ms": {"type": "integer"},
        },
    },
    "MemoryCleanupResult": {
        "type": "object",
        "description": "Result of memory cleanup",
        "properties": {
            "memories_removed": {"type": "integer"},
            "bytes_freed": {"type": "integer"},
            "duration_ms": {"type": "integer"},
        },
    },
    # Knowledge Mound schemas
    "KnowledgeNode": {
        "type": "object",
        "description": "A knowledge node in the mound",
        "properties": {
            "id": {"type": "string", "description": "Unique node ID"},
            "content": {"type": "string", "description": "Node content"},
            "source": {"type": "string", "description": "Knowledge source type"},
            "confidence": {"type": "number", "description": "Confidence score 0-1"},
            "created_at": {"type": "string", "format": "date-time"},
            "updated_at": {"type": "string", "format": "date-time"},
            "topics": {"type": "array", "items": {"type": "string"}},
            "metadata": {"type": "object", "additionalProperties": True},
        },
        "required": ["id", "content"],
    },
    "KnowledgeQueryResult": {
        "type": "object",
        "description": "Result of a knowledge query",
        "properties": {
            "items": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/KnowledgeNode"},
            },
            "total": {"type": "integer"},
            "query": {"type": "string"},
            "relevance_scores": {
                "type": "array",
                "items": {"type": "number"},
            },
        },
    },
    "KnowledgeStoreResult": {
        "type": "object",
        "description": "Result of storing knowledge",
        "properties": {
            "id": {"type": "string"},
            "success": {"type": "boolean"},
            "source": {"type": "string"},
            "timestamp": {"type": "string", "format": "date-time"},
        },
    },
    "KnowledgeFact": {
        "type": "object",
        "description": "A verified knowledge fact",
        "properties": {
            "id": {"type": "string"},
            "content": {"type": "string"},
            "source": {"type": "string"},
            "confidence": {"type": "number"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "contradictions": {"type": "array", "items": {"type": "string"}},
            "created_at": {"type": "string", "format": "date-time"},
            "verified_at": {"type": "string", "format": "date-time", "nullable": True},
        },
    },
    "KnowledgeFactList": {
        "type": "object",
        "description": "List of knowledge facts",
        "properties": {
            "facts": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/KnowledgeFact"},
            },
            "total": {"type": "integer"},
        },
    },
    "KnowledgeSearchResult": {
        "type": "object",
        "description": "A search result from the knowledge mound",
        "properties": {
            "node": {"$ref": "#/components/schemas/KnowledgeNode"},
            "score": {"type": "number"},
            "highlights": {"type": "array", "items": {"type": "string"}},
        },
    },
    "KnowledgeSearchResponse": {
        "type": "object",
        "description": "Search results from knowledge mound",
        "properties": {
            "results": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/KnowledgeSearchResult"},
            },
            "total": {"type": "integer"},
            "query": {"type": "string"},
            "search_time_ms": {"type": "integer"},
        },
    },
    "KnowledgeStats": {
        "type": "object",
        "description": "Knowledge mound statistics",
        "properties": {
            "total_nodes": {"type": "integer"},
            "total_facts": {"type": "integer"},
            "by_source": {"type": "object", "additionalProperties": {"type": "integer"}},
            "avg_confidence": {"type": "number"},
            "contradiction_rate": {"type": "number"},
        },
    },
    "KnowledgeQueryResponse": {
        "type": "object",
        "description": "Response from knowledge query",
        "properties": {
            "success": {"type": "boolean"},
            "result": {"$ref": "#/components/schemas/KnowledgeQueryResult"},
        },
    },
}


__all__ = ["MEMORY_SCHEMAS"]
