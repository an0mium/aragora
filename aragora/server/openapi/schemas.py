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
        "properties": {
            "error": {"type": "string", "description": "Error message"},
            "code": {"type": "string", "description": "Error code"},
            "trace_id": {"type": "string", "description": "Request trace ID for debugging"},
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
        "properties": {
            "task": {"type": "string"},
            "question": {"type": "string"},
            "agents": {"type": "array", "items": {"type": "string"}},
            "rounds": {"type": "integer"},
            "consensus": {"type": "string"},
            "context": {"type": "string"},
            "auto_select": {"type": "boolean"},
            "auto_select_config": {"type": "object"},
            "use_trending": {"type": "boolean"},
            "trending_category": {"type": "string"},
        },
    },
    "DebateCreateResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "debate_id": {"type": "string"},
            "status": {"$ref": "#/components/schemas/DebateStatus"},
            "task": {"type": "string"},
            "error": {"type": "string"},
        },
    },
    "Debate": {
        "type": "object",
        "properties": {
            "debate_id": {"type": "string"},
            "id": {"type": "string"},
            "slug": {"type": "string"},
            "task": {"type": "string"},
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
