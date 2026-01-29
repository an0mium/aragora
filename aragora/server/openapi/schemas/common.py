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
                "type": "string",
                "description": "Human-readable error message",
                "example": "Invalid request: missing required field 'task'",
            },
            "code": {
                "type": "string",
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
            "message": {"type": "string", "nullable": True},
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
            "timestamp": {"type": "string", "format": "date-time"},
            "checks": {"type": "object", "additionalProperties": {"type": "object"}},
            "response_time_ms": {"type": "number"},
        },
    },
}


__all__ = ["COMMON_SCHEMAS"]
