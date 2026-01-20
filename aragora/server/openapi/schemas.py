"""
OpenAPI Schema Definitions.

Contains common schema components used across all API endpoints.
"""

from typing import Any

# =============================================================================
# Common Schema Components
# =============================================================================

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
    "DebateStatus": {
        "type": "string",
        "enum": [
            "created",
            "starting",
            "pending",
            "running",
            "in_progress",
            "completed",
            "failed",
            "cancelled",
            "paused",
            "active",
            "concluded",
            "archived",
        ],
    },
    "ConsensusResult": {
        "type": "object",
        "properties": {
            "reached": {"type": "boolean"},
            "agreement": {"type": "number"},
            "confidence": {"type": "number"},
            "final_answer": {"type": "string"},
            "conclusion": {"type": "string"},
            "supporting_agents": {"type": "array", "items": {"type": "string"}},
            "dissenting_agents": {"type": "array", "items": {"type": "string"}},
        },
    },
    "DebateCreateRequest": {
        "type": "object",
        "description": "Request body for creating a new debate",
        "properties": {
            "task": {
                "type": "string",
                "description": "The topic or question for the debate",
                "example": "Should we adopt microservices architecture for our e-commerce platform?",
                "minLength": 10,
                "maxLength": 2000,
            },
            "question": {
                "type": "string",
                "description": "Alias for task (deprecated, use task instead)",
                "deprecated": True,
            },
            "agents": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of agent names to participate. If empty, auto_select is used.",
                "example": ["claude", "gpt-4", "gemini"],
                "minItems": 2,
                "maxItems": 8,
            },
            "rounds": {
                "type": "integer",
                "description": "Maximum number of debate rounds",
                "default": 3,
                "minimum": 1,
                "maximum": 10,
                "example": 3,
            },
            "consensus": {
                "type": "string",
                "description": "Consensus strategy to use",
                "enum": ["majority", "unanimous", "weighted", "semantic"],
                "default": "majority",
                "example": "majority",
            },
            "context": {
                "type": "string",
                "description": "Additional context or background information",
                "example": "We have 1M daily active users and need 99.9% uptime.",
                "maxLength": 10000,
            },
            "auto_select": {
                "type": "boolean",
                "description": "Automatically select optimal agents based on topic",
                "default": True,
            },
            "auto_select_config": {
                "type": "object",
                "description": "Configuration for auto-selection algorithm",
                "properties": {
                    "min_agents": {"type": "integer", "default": 3},
                    "max_agents": {"type": "integer", "default": 5},
                    "diversity_weight": {"type": "number", "default": 0.3},
                    "expertise_weight": {"type": "number", "default": 0.7},
                },
            },
            "use_trending": {
                "type": "boolean",
                "description": "Include trending context from news/social media",
                "default": False,
            },
            "trending_category": {
                "type": "string",
                "description": "Category filter for trending content",
                "enum": ["tech", "science", "politics", "business", "health"],
            },
        },
        "required": ["task"],
        "example": {
            "task": "Should we adopt microservices architecture for our e-commerce platform?",
            "agents": ["claude", "gpt-4", "gemini"],
            "rounds": 3,
            "consensus": "majority",
            "context": "We have 1M daily active users and need 99.9% uptime.",
        },
    },
    "DebateCreateResponse": {
        "type": "object",
        "description": "Response when a debate is successfully created",
        "properties": {
            "success": {
                "type": "boolean",
                "description": "Whether the debate was created successfully",
                "example": True,
            },
            "debate_id": {
                "type": "string",
                "description": "Unique identifier for the created debate",
                "example": "deb_abc123xyz",
            },
            "status": {
                "$ref": "#/components/schemas/DebateStatus",
                "description": "Current status of the debate",
            },
            "task": {
                "type": "string",
                "description": "The debate topic (echoed back)",
                "example": "Should we adopt microservices architecture?",
            },
            "agents": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Agents participating in the debate",
                "example": ["claude", "gpt-4", "gemini"],
            },
            "websocket_url": {
                "type": "string",
                "description": "WebSocket URL to stream debate progress",
                "example": "wss://api.aragora.ai/ws/debates/deb_abc123xyz",
            },
            "estimated_duration": {
                "type": "integer",
                "description": "Estimated debate duration in seconds",
                "example": 120,
            },
            "error": {
                "type": "string",
                "description": "Error message if success is false",
            },
        },
        "required": ["success"],
        "example": {
            "success": True,
            "debate_id": "deb_abc123xyz",
            "status": "running",
            "task": "Should we adopt microservices architecture?",
            "agents": ["claude", "gpt-4", "gemini"],
            "websocket_url": "wss://api.aragora.ai/ws/debates/deb_abc123xyz",
            "estimated_duration": 120,
        },
    },
    "Debate": {
        "type": "object",
        "properties": {
            "debate_id": {"type": "string"},
            "id": {"type": "string"},
            "slug": {"type": "string"},
            "task": {"type": "string"},
            "topic": {"type": "string", "description": "Alias for task"},
            "context": {"type": "string"},
            "status": {"$ref": "#/components/schemas/DebateStatus"},
            "outcome": {"type": "string"},
            "final_answer": {"type": "string"},
            "consensus": {"$ref": "#/components/schemas/ConsensusResult"},
            "consensus_proof": {"type": "object"},
            "consensus_reached": {"type": "boolean"},
            "confidence": {"type": "number"},
            "rounds_used": {"type": "integer"},
            "duration_seconds": {"type": "number"},
            "agents": {"type": "array", "items": {"type": "string"}},
            "rounds": {"type": "array", "items": {"$ref": "#/components/schemas/Round"}},
            "created_at": {"type": "string", "format": "date-time"},
            "completed_at": {"type": "string", "format": "date-time"},
            "metadata": {"type": "object"},
        },
    },
    "Message": {
        "type": "object",
        "properties": {
            "role": {"type": "string", "enum": ["system", "user", "assistant"]},
            "content": {"type": "string"},
            "agent": {"type": "string"},
            "agent_id": {"type": "string"},
            "round": {"type": "integer"},
            "timestamp": {"type": "string", "format": "date-time"},
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
    "Consensus": {
        "type": "object",
        "properties": {
            "reached": {"type": "boolean"},
            "topic": {"type": "string"},
            "verdict": {"type": "string"},
            "confidence": {"type": "number"},
            "participating_agents": {"type": "array", "items": {"type": "string"}},
        },
    },
    "Calibration": {
        "type": "object",
        "properties": {
            "agent": {"type": "string"},
            "score": {"type": "number", "description": "Calibration score (0-1)"},
            "bucket_stats": {"type": "array", "items": {"type": "object"}},
            "overconfidence_index": {"type": "number"},
        },
    },
    "Relationship": {
        "type": "object",
        "properties": {
            "agent_a": {"type": "string"},
            "agent_b": {"type": "string"},
            "alliance_score": {"type": "number"},
            "rivalry_score": {"type": "number"},
            "total_interactions": {"type": "integer"},
        },
    },
    "OAuthProvider": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Provider identifier (e.g., 'google', 'github')",
            },
            "name": {"type": "string", "description": "Display name for the provider"},
        },
        "required": ["id", "name"],
    },
    "OAuthProviders": {
        "type": "object",
        "properties": {
            "providers": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/OAuthProvider"},
                "description": "List of available OAuth providers",
            },
        },
        "required": ["providers"],
    },
    "Round": {
        "type": "object",
        "properties": {
            "round_number": {"type": "integer", "description": "Round number (1-indexed)"},
            "messages": {"type": "array", "items": {"$ref": "#/components/schemas/Message"}},
            "votes": {"type": "object", "description": "Agent votes for this round"},
            "summary": {"type": "string", "description": "Round summary"},
        },
    },
    "Workspace": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Workspace ID"},
            "organization_id": {"type": "string", "description": "Parent organization ID"},
            "name": {"type": "string", "description": "Workspace name"},
            "created_at": {"type": "string", "format": "date-time"},
            "created_by": {"type": "string"},
            "encrypted": {"type": "boolean", "description": "Whether workspace data is encrypted"},
            "retention_days": {"type": "integer", "description": "Data retention period in days"},
            "sensitivity_level": {"type": "string", "description": "Data sensitivity level"},
            "document_count": {"type": "integer"},
            "storage_bytes": {"type": "integer"},
        },
        "required": ["id", "organization_id", "name"],
    },
    "WorkspaceList": {
        "type": "object",
        "properties": {
            "workspaces": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/Workspace"},
            },
            "total": {"type": "integer"},
        },
        "required": ["workspaces", "total"],
    },
    # Retention Policies
    "RetentionPolicy": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "retention_days": {"type": "integer"},
            "data_types": {"type": "array", "items": {"type": "string"}},
            "enabled": {"type": "boolean"},
            "created_at": {"type": "string", "format": "date-time"},
        },
    },
    "RetentionPolicyList": {
        "type": "object",
        "properties": {
            "policies": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/RetentionPolicy"},
            },
            "total": {"type": "integer"},
        },
    },
    # Workflow schemas
    "StepDefinition": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "type": {"type": "string"},
            "config": {"type": "object"},
            "depends_on": {"type": "array", "items": {"type": "string"}},
        },
    },
    "TransitionRule": {
        "type": "object",
        "properties": {
            "from_step": {"type": "string"},
            "to_step": {"type": "string"},
            "condition": {"type": "string"},
        },
    },
    "Workflow": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "description": {"type": "string"},
            "version": {"type": "string"},
            "status": {"type": "string"},
            "steps": {"type": "array", "items": {"$ref": "#/components/schemas/StepDefinition"}},
            "transitions": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/TransitionRule"},
            },
            "created_at": {"type": "string", "format": "date-time"},
            "updated_at": {"type": "string", "format": "date-time"},
        },
    },
    "WorkflowList": {
        "type": "object",
        "properties": {
            "workflows": {"type": "array", "items": {"$ref": "#/components/schemas/Workflow"}},
            "total": {"type": "integer"},
        },
    },
    "WorkflowUpdate": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "steps": {"type": "array", "items": {"$ref": "#/components/schemas/StepDefinition"}},
            "transitions": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/TransitionRule"},
            },
        },
    },
    "WorkflowTemplate": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "description": {"type": "string"},
            "category": {"type": "string"},
            "steps": {"type": "array", "items": {"$ref": "#/components/schemas/StepDefinition"}},
            "parameters": {"type": "object"},
        },
    },
    "WorkflowTemplateList": {
        "type": "object",
        "properties": {
            "templates": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/WorkflowTemplate"},
            },
            "total": {"type": "integer"},
        },
    },
    "ExecutionList": {
        "type": "object",
        "properties": {
            "executions": {"type": "array", "items": {"type": "object"}},
            "total": {"type": "integer"},
        },
    },
}


# =============================================================================
# Response Helpers
# =============================================================================


def ok_response(description: str, schema_ref: str | None = None) -> dict:
    """Create a successful response definition."""
    resp: dict = {"description": description}
    if schema_ref:
        resp["content"] = {
            "application/json": {"schema": {"$ref": f"#/components/schemas/{schema_ref}"}}
        }
    return resp


def array_response(description: str, schema_ref: str) -> dict:
    """Create an array response definition."""
    return {
        "description": description,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {"$ref": f"#/components/schemas/{schema_ref}"},
                        },
                        "total": {"type": "integer"},
                    },
                }
            }
        },
    }


def error_response(status: str, description: str) -> dict:
    """Create an error response definition."""
    return {
        "description": description,
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}},
    }


# Standard error responses used across endpoints
STANDARD_ERRORS = {
    "400": error_response("400", "Bad request"),
    "401": error_response("401", "Unauthorized"),
    "404": error_response("404", "Not found"),
    "402": error_response("402", "Quota exceeded"),
    "429": error_response("429", "Rate limited"),
    "500": error_response("500", "Server error"),
}


__all__ = [
    "COMMON_SCHEMAS",
    "STANDARD_ERRORS",
    "ok_response",
    "array_response",
    "error_response",
]
