"""
Common OpenAPI Schema Definitions.

Base schemas used across all API endpoints.
"""

from typing import Any

COMMON_SCHEMAS: dict[str, Any] = {
    "Error": {
        "type": "object",
        "description": "Standard error response format",
        "properties": {
            "error": {
                "type": ["string", "null"],
                "description": "Human-readable error message",
                "example": "Invalid request: missing required field 'task'",
            },
            "code": {
                "type": ["string", "null"],
                "description": "Machine-readable error code for programmatic handling",
                "enum": [
                    "INVALID_JSON",
                    "MISSING_FIELD",
                    "INVALID_VALUE",
                    "AUTH_REQUIRED",
                    "INVALID_TOKEN",
                    "FORBIDDEN",
                    "NOT_OWNER",
                    "NOT_FOUND",
                    "QUOTA_EXCEEDED",
                    "RATE_LIMITED",
                    "INTERNAL_ERROR",
                    "SERVICE_UNAVAILABLE",
                    "AGENT_TIMEOUT",
                    "CONSENSUS_FAILED",
                ],
                "example": "MISSING_FIELD",
            },
            "trace_id": {
                "type": "string",
                "description": "Unique request ID for debugging and support",
                "example": "req_abc123xyz789",
            },
            "field": {
                "type": "string",
                "description": "Name of the field that caused the error (for validation errors)",
                "example": "task",
            },
            "resource_type": {
                "type": "string",
                "description": "Type of resource involved in the error",
                "example": "debate",
            },
            "resource_id": {
                "type": "string",
                "description": "ID of the resource involved in the error",
                "example": "deb_abc123",
            },
            "limit": {
                "type": "integer",
                "description": "The limit that was exceeded (for quota/rate errors)",
                "example": 60,
            },
            "retry_after": {
                "type": "integer",
                "description": "Seconds to wait before retrying (for rate limit errors)",
                "example": 45,
            },
            "resets_at": {
                "type": "string",
                "format": "date-time",
                "description": "When the quota/rate limit resets",
                "example": "2024-01-16T00:00:00Z",
            },
            "upgrade_url": {
                "type": "string",
                "format": "uri",
                "description": "URL to upgrade plan (for quota errors)",
                "example": "https://aragora.ai/pricing",
            },
            "support_url": {
                "type": "string",
                "format": "uri",
                "description": "URL for support/issue reporting",
                "example": "https://github.com/anthropics/aragora/issues",
            },
        },
        "required": ["error"],
    },
    "PaginatedResponse": {
        "type": "object",
        "properties": {
            "total": {"type": "integer", "description": "Total items available"},
            "offset": {"type": "integer", "description": "Current offset"},
            "limit": {"type": "integer", "description": "Page size"},
            "has_more": {"type": "boolean", "description": "More items available"},
        },
    },
    "StandardSuccessResponse": {
        "type": "object",
        "description": "Generic success response wrapper",
        "properties": {
            "success": {"type": "boolean"},
            "data": {"type": "object", "additionalProperties": True},
            "message": {"type": ["string", "null"]},
        },
        "required": ["success"],
    },
    "Agent": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Agent name"},
            "elo": {"type": "number", "description": "ELO rating"},
            "matches": {"type": "integer", "description": "Total matches played"},
            "wins": {"type": "integer", "description": "Total wins"},
            "losses": {"type": "integer", "description": "Total losses"},
            "calibration_score": {"type": "number", "description": "Calibration accuracy (0-1)"},
        },
    },
    "HealthCheck": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["healthy", "degraded", "unhealthy"]},
            "version": {"type": "string"},
            "uptime_seconds": {"type": "number", "description": "Server uptime in seconds"},
            "demo_mode": {"type": "boolean", "description": "Whether the server is in demo mode"},
            "timestamp": {"type": "string", "format": "date-time"},
            "checks": {"type": "object", "additionalProperties": {"type": "object"}},
            "response_time_ms": {"type": "number"},
        },
    },
    "StarterTemplate": {
        "type": "object",
        "description": "Template for onboarding debates",
        "properties": {
            "id": {"type": "string", "description": "Unique template identifier"},
            "name": {"type": "string", "description": "Display name"},
            "description": {"type": "string", "description": "Template description"},
            "use_cases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Applicable use cases",
            },
            "agents_count": {"type": "integer", "description": "Number of agents"},
            "rounds": {"type": "integer", "description": "Number of debate rounds"},
            "estimated_minutes": {"type": "integer", "description": "Estimated duration"},
            "example_prompt": {"type": "string", "description": "Example prompt"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Categorization tags",
            },
            "difficulty": {"type": "string", "description": "Difficulty level"},
        },
        "required": [
            "id",
            "name",
            "description",
            "use_cases",
            "agents_count",
            "rounds",
            "estimated_minutes",
            "example_prompt",
        ],
    },
    "GauntletRun": {
        "type": "object",
        "description": "Gauntlet verification run result",
        "properties": {
            "id": {"type": "string", "description": "Run identifier"},
            "status": {
                "type": "string",
                "enum": ["pending", "running", "completed", "failed"],
                "description": "Run status",
            },
            "verdict": {"type": ["string", "null"], "description": "Final verdict"},
            "confidence": {"type": "number", "description": "Confidence score"},
            "findings": {
                "type": "array",
                "items": {"type": "object"},
                "description": "List of findings",
            },
            "metadata": {"type": "object", "description": "Additional metadata"},
            "created_at": {
                "type": "string",
                "format": "date-time",
                "description": "Creation timestamp",
            },
            "started_at": {
                "type": ["string", "null"],
                "format": "date-time",
                "description": "Start timestamp",
            },
            "completed_at": {
                "type": ["string", "null"],
                "format": "date-time",
                "description": "Completion timestamp",
            },
        },
        "required": ["id", "status"],
    },
    "GauntletComparison": {
        "type": "object",
        "description": "Comparison between two gauntlet runs",
        "properties": {
            "run_a": {"$ref": "#/components/schemas/GauntletRun"},
            "run_b": {"$ref": "#/components/schemas/GauntletRun"},
            "diff": {"type": "object", "description": "Differences between runs"},
            "similarity_score": {"type": "number", "description": "Similarity percentage"},
            "comparison_notes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Notes about differences",
            },
        },
    },
    "Policy": {
        "type": "object",
        "description": "Policy definition",
        "properties": {
            "id": {"type": "string", "description": "Policy identifier"},
            "name": {"type": "string", "description": "Policy name"},
            "description": {"type": "string", "description": "Policy description"},
            "type": {"type": "string", "description": "Policy type"},
            "rules": {"type": "array", "items": {"type": "object"}, "description": "Policy rules"},
            "enabled": {"type": "boolean", "description": "Whether policy is enabled"},
            "priority": {"type": "integer", "description": "Execution priority"},
            "created_at": {
                "type": "string",
                "format": "date-time",
                "description": "Creation timestamp",
            },
            "updated_at": {
                "type": "string",
                "format": "date-time",
                "description": "Last update timestamp",
            },
        },
        "required": ["id", "name"],
    },
}


__all__ = ["COMMON_SCHEMAS"]
