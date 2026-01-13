"""
OpenAPI Schema Generator for Aragora API.

Generates OpenAPI 3.0 specification for all API endpoints.
Endpoints are organized by tag/category for clear documentation.

Usage:
    from aragora.server.openapi import generate_openapi_schema, save_openapi_schema

    # Get schema as dict
    schema = generate_openapi_schema()

    # Save to file
    path, count = save_openapi_schema("docs/api/openapi.json")
"""

import json
from pathlib import Path
from typing import Any

# API version
API_VERSION = "1.0.0"

# =============================================================================
# Common Schema Components
# =============================================================================

COMMON_SCHEMAS = {
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
# Endpoint Definitions by Category
# =============================================================================


# Helper functions for common response patterns
def _ok_response(description: str, schema_ref: str | None = None) -> dict:
    resp: dict = {"description": description}
    if schema_ref:
        resp["content"] = {
            "application/json": {"schema": {"$ref": f"#/components/schemas/{schema_ref}"}}
        }
    return resp


def _array_response(description: str, schema_ref: str):
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


def _error_response(status: str, description: str):
    return {
        "description": description,
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}},
    }


STANDARD_ERRORS = {
    "400": _error_response("400", "Bad request"),
    "401": _error_response("401", "Unauthorized"),
    "404": _error_response("404", "Not found"),
    "402": _error_response("402", "Quota exceeded"),
    "429": _error_response("429", "Rate limited"),
    "500": _error_response("500", "Server error"),
}

# =============================================================================
# System Endpoints
# =============================================================================
SYSTEM_ENDPOINTS = {
    "/api/health": {
        "get": {
            "tags": ["System"],
            "summary": "Health check",
            "description": "Get system health status for load balancers and monitoring. Returns 200 when healthy, 503 when degraded.",
            "responses": {
                "200": _ok_response("System healthy", "HealthCheck"),
                "503": {"description": "System degraded"},
            },
        },
    },
    "/api/health/detailed": {
        "get": {
            "tags": ["System"],
            "summary": "Detailed health check",
            "description": "Get detailed health status with component checks, observer metrics, memory stats",
            "responses": {"200": _ok_response("Detailed health information")},
        },
    },
    "/api/nomic/state": {
        "get": {
            "tags": ["System"],
            "summary": "Get nomic loop state",
            "description": "Get current state of the nomic self-improvement loop",
            "responses": {"200": _ok_response("Nomic state")},
        },
    },
    "/api/nomic/health": {
        "get": {
            "tags": ["System"],
            "summary": "Nomic loop health",
            "description": "Get nomic loop health with stall detection",
            "responses": {"200": _ok_response("Nomic health status")},
        },
    },
    "/api/nomic/log": {
        "get": {
            "tags": ["System"],
            "summary": "Get nomic logs",
            "description": "Get recent nomic loop log lines",
            "parameters": [
                {
                    "name": "lines",
                    "in": "query",
                    "schema": {"type": "integer", "default": 100, "maximum": 1000},
                },
            ],
            "responses": {"200": _ok_response("Log lines")},
        },
    },
    "/api/nomic/risk-register": {
        "get": {
            "tags": ["System"],
            "summary": "Risk register",
            "description": "Get risk register entries from nomic loop execution",
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 50, "maximum": 200},
                },
            ],
            "responses": {"200": _ok_response("Risk entries")},
        },
    },
    "/api/modes": {
        "get": {
            "tags": ["System"],
            "summary": "List operational modes",
            "description": "Get available operational modes (builtin + custom)",
            "responses": {"200": _ok_response("Available modes")},
        },
    },
    "/api/history/cycles": {
        "get": {
            "tags": ["System"],
            "summary": "Cycle history",
            "parameters": [
                {"name": "loop_id", "in": "query", "schema": {"type": "string"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 50}},
            ],
            "responses": {"200": _ok_response("Cycle history")},
        },
    },
    "/api/history/events": {
        "get": {
            "tags": ["System"],
            "summary": "Event history",
            "parameters": [
                {"name": "loop_id", "in": "query", "schema": {"type": "string"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 100}},
            ],
            "responses": {"200": _ok_response("Event history")},
        },
    },
    "/api/history/debates": {
        "get": {
            "tags": ["System"],
            "summary": "Debate history",
            "parameters": [
                {"name": "loop_id", "in": "query", "schema": {"type": "string"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 50}},
            ],
            "responses": {"200": _ok_response("Debate history")},
        },
    },
    "/api/history/summary": {
        "get": {
            "tags": ["System"],
            "summary": "History summary",
            "parameters": [{"name": "loop_id", "in": "query", "schema": {"type": "string"}}],
            "responses": {"200": _ok_response("Summary statistics")},
        },
    },
    "/api/system/maintenance": {
        "get": {
            "tags": ["System"],
            "summary": "Run database maintenance",
            "parameters": [
                {
                    "name": "task",
                    "in": "query",
                    "schema": {
                        "type": "string",
                        "enum": ["status", "vacuum", "analyze", "checkpoint", "full"],
                        "default": "status",
                    },
                },
            ],
            "responses": {"200": _ok_response("Maintenance results")},
        },
    },
    "/api/openapi": {
        "get": {
            "tags": ["System"],
            "summary": "OpenAPI specification",
            "description": "Get OpenAPI 3.0 schema for this API",
            "responses": {
                "200": {"description": "OpenAPI schema", "content": {"application/json": {}}}
            },
        },
    },
}

# =============================================================================
# Agent Endpoints
# =============================================================================
AGENT_ENDPOINTS = {
    "/api/agents": {
        "get": {
            "tags": ["Agents"],
            "summary": "List all agents",
            "description": "Get list of all known agents with optional stats",
            "parameters": [
                {
                    "name": "include_stats",
                    "in": "query",
                    "schema": {"type": "boolean", "default": False},
                },
            ],
            "responses": {"200": _array_response("List of agents", "Agent")},
        },
    },
    "/api/leaderboard": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent leaderboard",
            "description": "Get agents ranked by ELO rating",
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 20, "maximum": 100},
                },
                {
                    "name": "domain",
                    "in": "query",
                    "schema": {"type": "string"},
                    "description": "Filter by expertise domain",
                },
            ],
            "responses": {"200": _ok_response("Agent rankings")},
        },
    },
    "/api/rankings": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent rankings",
            "description": "Alternative endpoint for agent rankings",
            "parameters": [
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}},
            ],
            "responses": {"200": _ok_response("Agent rankings")},
        },
    },
    "/api/leaderboard-view": {
        "get": {
            "tags": ["Agents"],
            "summary": "Leaderboard view data",
            "description": "Pre-formatted leaderboard data for frontend display",
            "parameters": [
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 10}},
                {"name": "domain", "in": "query", "schema": {"type": "string"}},
                {"name": "loop_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {"200": _ok_response("Leaderboard view")},
        },
    },
    "/api/agent/{name}/profile": {
        "get": {
            "tags": ["Agents"],
            "summary": "Get agent profile",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Agent profile", "Agent"), **STANDARD_ERRORS},
        },
    },
    "/api/agent/{name}/history": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent match history",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}},
            ],
            "responses": {"200": _ok_response("Match history")},
        },
    },
    "/api/agent/{name}/calibration": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent calibration data",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Calibration data", "Calibration")},
        },
    },
    "/api/agent/{name}/calibration-curve": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent calibration curve",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Calibration curve data")},
        },
    },
    "/api/agent/{name}/calibration-summary": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent calibration summary",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Calibration summary")},
        },
    },
    "/api/agent/{name}/consistency": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent consistency score",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Consistency metrics")},
        },
    },
    "/api/agent/{name}/flips": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent position flips",
            "description": "Get instances where agent changed positions",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}},
            ],
            "responses": {"200": _ok_response("Position flips")},
        },
    },
    "/api/agent/{name}/network": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent relationship network",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Relationship network")},
        },
    },
    "/api/agent/{name}/rivals": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent rivals",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Rival agents")},
        },
    },
    "/api/agent/{name}/allies": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent allies",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Allied agents")},
        },
    },
    "/api/agent/{name}/moments": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent moments",
            "description": "Get significant moments for this agent",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Agent moments")},
        },
    },
    "/api/agent/{name}/reputation": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent reputation",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Reputation data")},
        },
    },
    "/api/agent/{name}/persona": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent persona",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Persona data")},
        },
    },
    "/api/agent/{name}/grounded-persona": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent grounded persona",
            "description": "Get persona derived from debate performance",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Grounded persona")},
        },
    },
    "/api/agent/{name}/identity-prompt": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent identity prompt",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Identity prompt")},
        },
    },
    "/api/agent/{name}/performance": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent performance",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Performance metrics")},
        },
    },
    "/api/agent/{name}/domains": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent expertise domains",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Domain expertise")},
        },
    },
    "/api/agent/{name}/accuracy": {
        "get": {
            "tags": ["Agents"],
            "summary": "Agent accuracy",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Accuracy metrics")},
        },
    },
    "/api/agent/compare": {
        "get": {
            "tags": ["Agents"],
            "summary": "Compare agents",
            "description": "Compare two agents side-by-side",
            "parameters": [
                {"name": "agent_a", "in": "query", "required": True, "schema": {"type": "string"}},
                {"name": "agent_b", "in": "query", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {"200": _ok_response("Comparison data")},
        },
    },
    "/api/matches/recent": {
        "get": {
            "tags": ["Agents"],
            "summary": "Recent matches",
            "parameters": [
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}}
            ],
            "responses": {"200": _ok_response("Recent matches")},
        },
    },
    "/api/calibration/leaderboard": {
        "get": {
            "tags": ["Agents"],
            "summary": "Calibration leaderboard",
            "description": "Get agents ranked by calibration score",
            "parameters": [
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}}
            ],
            "responses": {"200": _ok_response("Calibration rankings")},
        },
    },
    "/api/personas": {
        "get": {
            "tags": ["Agents"],
            "summary": "List all personas",
            "responses": {"200": _ok_response("All personas")},
        },
    },
}

# =============================================================================
# Debate Endpoints
# =============================================================================
DEBATE_ENDPOINTS = {
    "/api/debates": {
        "get": {
            "tags": ["Debates"],
            "summary": "List debates",
            "description": "Get list of all debates (requires auth)",
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 20, "maximum": 100},
                },
                {"name": "offset", "in": "query", "schema": {"type": "integer", "default": 0}},
            ],
            "responses": {"200": _array_response("List of debates", "Debate")},
            "security": [{"bearerAuth": []}],
        },
        "post": {
            "tags": ["Debates"],
            "summary": "Create a new debate",
            "description": "Start a new multi-agent debate on a given topic",
            "security": [{"bearerAuth": []}],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/DebateCreateRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response("Debate created successfully", "DebateCreateResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "402": STANDARD_ERRORS["402"],
                "429": STANDARD_ERRORS["429"],
            },
        },
    },
    "/api/debate": {
        "post": {
            "tags": ["Debates"],
            "summary": "Create a new debate (deprecated)",
            "description": "Deprecated. Use POST /api/debates instead.",
            "deprecated": True,
            "security": [{"bearerAuth": []}],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/DebateCreateRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response("Debate created successfully", "DebateCreateResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
            },
        },
    },
    "/api/debates/{id}": {
        "get": {
            "tags": ["Debates"],
            "summary": "Get debate by ID",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Debate details", "Debate"),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/debates/slug/{slug}": {
        "get": {
            "tags": ["Debates"],
            "summary": "Get debate by slug",
            "parameters": [
                {"name": "slug", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Debate details", "Debate"),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/debates/{id}/messages": {
        "get": {
            "tags": ["Debates"],
            "summary": "Get debate messages",
            "description": "Get paginated message history for a debate",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 50, "maximum": 200},
                },
                {"name": "offset", "in": "query", "schema": {"type": "integer", "default": 0}},
            ],
            "responses": {"200": _ok_response("Paginated messages")},
        },
    },
    "/api/debates/{id}/convergence": {
        "get": {
            "tags": ["Debates"],
            "summary": "Get convergence status",
            "description": "Check if debate has reached semantic convergence",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Convergence status")},
        },
    },
    "/api/debates/{id}/citations": {
        "get": {
            "tags": ["Debates"],
            "summary": "Get evidence citations",
            "description": "Get grounded verdict with evidence citations",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Citations and grounding score")},
        },
    },
    "/api/debates/{id}/evidence": {
        "get": {
            "tags": ["Debates"],
            "summary": "Get debate evidence",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Evidence data")},
        },
    },
    "/api/debates/{id}/impasse": {
        "get": {
            "tags": ["Debates"],
            "summary": "Get impasse status",
            "description": "Check if debate reached an impasse",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Impasse status")},
        },
    },
    "/api/debates/{id}/meta-critique": {
        "get": {
            "tags": ["Debates"],
            "summary": "Get meta-critique",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Meta-critique data")},
        },
    },
    "/api/debates/{id}/graph/stats": {
        "get": {
            "tags": ["Debates"],
            "summary": "Get debate graph stats",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Graph statistics")},
        },
    },
    "/api/debates/{id}/fork": {
        "post": {
            "tags": ["Debates"],
            "summary": "Fork debate",
            "description": "Create a counterfactual branch from a specific round",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "branch_point": {
                                    "type": "integer",
                                    "description": "Round to branch from",
                                },
                                "new_premise": {
                                    "type": "string",
                                    "description": "New premise for fork",
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "201": _ok_response("Forked debate created"),
                "400": STANDARD_ERRORS["400"],
            },
        },
    },
    "/api/debates/{id}/export/{format}": {
        "get": {
            "tags": ["Debates"],
            "summary": "Export debate",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {
                    "name": "format",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string", "enum": ["json", "markdown", "html", "pdf"]},
                },
            ],
            "responses": {"200": _ok_response("Exported debate")},
        },
    },
    "/api/debates/{id}/broadcast": {
        "post": {
            "tags": ["Debates"],
            "summary": "Generate debate broadcast",
            "description": "Generate audio/video broadcast of debate",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "format": {"type": "string", "enum": ["audio", "video"]},
                                "voices": {"type": "object"},
                            },
                        }
                    }
                },
            },
            "responses": {"202": _ok_response("Broadcast generation started")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/debates/{id}/publish/twitter": {
        "post": {
            "tags": ["Social"],
            "summary": "Publish to Twitter",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Published to Twitter")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/debates/{id}/publish/youtube": {
        "post": {
            "tags": ["Social"],
            "summary": "Publish to YouTube",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Published to YouTube")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/debates/{id}/red-team": {
        "get": {
            "tags": ["Auditing"],
            "summary": "Get red team results",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Red team results")},
        },
    },
    "/api/search": {
        "get": {
            "tags": ["Debates"],
            "summary": "Cross-debate search",
            "parameters": [
                {"name": "q", "in": "query", "required": True, "schema": {"type": "string"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}},
            ],
            "responses": {"200": _ok_response("Search results")},
        },
    },
    "/api/dashboard/debates": {
        "get": {
            "tags": ["Debates"],
            "summary": "Dashboard debates",
            "description": "Get debates formatted for dashboard display",
            "parameters": [
                {"name": "domain", "in": "query", "schema": {"type": "string"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 10}},
                {"name": "hours", "in": "query", "schema": {"type": "integer", "default": 24}},
            ],
            "responses": {"200": _ok_response("Dashboard data")},
        },
    },
}

# =============================================================================
# Analytics & Insights Endpoints
# =============================================================================
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

# =============================================================================
# Consensus Endpoints
# =============================================================================
CONSENSUS_ENDPOINTS = {
    "/api/consensus/similar": {
        "get": {
            "tags": ["Consensus"],
            "summary": "Similar debates",
            "description": "Find debates similar to a given topic",
            "parameters": [
                {"name": "topic", "in": "query", "required": True, "schema": {"type": "string"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 10}},
            ],
            "responses": {"200": _ok_response("Similar debates")},
        },
    },
    "/api/consensus/settled": {
        "get": {
            "tags": ["Consensus"],
            "summary": "Settled questions",
            "parameters": [
                {"name": "threshold", "in": "query", "schema": {"type": "number", "default": 0.8}},
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}},
            ],
            "responses": {"200": _ok_response("Settled questions")},
        },
    },
    "/api/consensus/stats": {
        "get": {
            "tags": ["Consensus"],
            "summary": "Consensus statistics",
            "responses": {"200": _ok_response("Consensus stats")},
        },
    },
    "/api/consensus/dissents": {
        "get": {
            "tags": ["Consensus"],
            "summary": "Dissenting views",
            "parameters": [
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}}
            ],
            "responses": {"200": _ok_response("Dissenting views")},
        },
    },
    "/api/consensus/contrarian-views": {
        "get": {
            "tags": ["Consensus"],
            "summary": "Contrarian views",
            "responses": {"200": _ok_response("Contrarian views")},
        },
    },
    "/api/consensus/risk-warnings": {
        "get": {
            "tags": ["Consensus"],
            "summary": "Risk warnings",
            "responses": {"200": _ok_response("Risk warnings")},
        },
    },
    "/api/consensus/domain/{domain}": {
        "get": {
            "tags": ["Consensus"],
            "summary": "Domain consensus",
            "parameters": [
                {"name": "domain", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Domain consensus data")},
        },
    },
}

# =============================================================================
# Relationship Endpoints
# =============================================================================
RELATIONSHIP_ENDPOINTS = {
    "/api/relationships/summary": {
        "get": {
            "tags": ["Relationships"],
            "summary": "Relationship summary",
            "responses": {"200": _ok_response("Relationship summary")},
        },
    },
    "/api/relationships/graph": {
        "get": {
            "tags": ["Relationships"],
            "summary": "Relationship graph",
            "description": "Get graph data for agent relationships",
            "responses": {"200": _ok_response("Graph data")},
        },
    },
    "/api/relationships/stats": {
        "get": {
            "tags": ["Relationships"],
            "summary": "Relationship statistics",
            "responses": {"200": _ok_response("Relationship stats")},
        },
    },
    "/api/relationship/{agent_a}/{agent_b}": {
        "get": {
            "tags": ["Relationships"],
            "summary": "Get relationship between two agents",
            "parameters": [
                {"name": "agent_a", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "agent_b", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {"200": _ok_response("Relationship data", "Relationship")},
        },
    },
}

# =============================================================================
# Memory Endpoints
# =============================================================================
MEMORY_ENDPOINTS = {
    "/api/memory/continuum/retrieve": {
        "get": {
            "tags": ["Memory"],
            "summary": "Retrieve memories",
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
            "description": "Trigger memory consolidation across tiers",
            "responses": {"200": _ok_response("Consolidation result")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/memory/continuum/cleanup": {
        "post": {
            "tags": ["Memory"],
            "summary": "Cleanup memories",
            "description": "Remove expired memories",
            "responses": {"200": _ok_response("Cleanup result")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/memory/tier-stats": {
        "get": {
            "tags": ["Memory"],
            "summary": "Memory tier statistics",
            "responses": {"200": _ok_response("Tier stats")},
        },
    },
    "/api/memory/archive-stats": {
        "get": {
            "tags": ["Memory"],
            "summary": "Archive statistics",
            "responses": {"200": _ok_response("Archive stats")},
        },
    },
}

# =============================================================================
# Belief Network Endpoints
# =============================================================================
BELIEF_ENDPOINTS = {
    "/api/belief-network/{debate_id}/cruxes": {
        "get": {
            "tags": ["Belief"],
            "summary": "Get debate cruxes",
            "parameters": [
                {"name": "debate_id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Debate cruxes")},
        },
    },
    "/api/belief-network/{debate_id}/load-bearing-claims": {
        "get": {
            "tags": ["Belief"],
            "summary": "Get load-bearing claims",
            "parameters": [
                {"name": "debate_id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Load-bearing claims")},
        },
    },
    "/api/debate/{debate_id}/graph-stats": {
        "get": {
            "tags": ["Belief"],
            "summary": "Get debate graph stats",
            "parameters": [
                {"name": "debate_id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Graph statistics")},
        },
    },
}

# =============================================================================
# Pulse & Trending Endpoints
# =============================================================================
PULSE_ENDPOINTS = {
    "/api/pulse/trending": {
        "get": {
            "tags": ["Pulse"],
            "summary": "Trending topics",
            "description": "Get current trending debate topics",
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 10, "maximum": 50},
                }
            ],
            "responses": {"200": _ok_response("Trending topics")},
        },
    },
    "/api/pulse/suggest": {
        "get": {
            "tags": ["Pulse"],
            "summary": "Suggest debate topic",
            "description": "Get AI-suggested debate topic based on trends",
            "parameters": [{"name": "category", "in": "query", "schema": {"type": "string"}}],
            "responses": {"200": _ok_response("Suggested topic")},
        },
    },
}

# =============================================================================
# Monitoring & Metrics Endpoints
# =============================================================================
METRICS_ENDPOINTS = {
    "/api/metrics": {
        "get": {
            "tags": ["Monitoring"],
            "summary": "System metrics",
            "responses": {"200": _ok_response("Metrics data")},
        },
    },
    "/api/metrics/health": {
        "get": {
            "tags": ["Monitoring"],
            "summary": "Metrics health",
            "responses": {"200": _ok_response("Metrics health")},
        },
    },
    "/api/metrics/cache": {
        "get": {
            "tags": ["Monitoring"],
            "summary": "Cache metrics",
            "responses": {"200": _ok_response("Cache metrics")},
        },
    },
    "/api/metrics/system": {
        "get": {
            "tags": ["Monitoring"],
            "summary": "System metrics",
            "responses": {"200": _ok_response("System metrics")},
        },
    },
    "/metrics": {
        "get": {
            "tags": ["Monitoring"],
            "summary": "Prometheus metrics",
            "description": "Metrics in Prometheus format",
            "responses": {
                "200": {
                    "description": "Prometheus-formatted metrics",
                    "content": {"text/plain": {}},
                }
            },
        },
    },
}

# =============================================================================
# Verification & Auditing Endpoints
# =============================================================================
VERIFICATION_ENDPOINTS = {
    "/api/verification/status": {
        "get": {
            "tags": ["Verification"],
            "summary": "Verification status",
            "responses": {"200": _ok_response("Verification status")},
        },
    },
    "/api/verification/formal-verify": {
        "post": {
            "tags": ["Verification"],
            "summary": "Formal verification",
            "description": "Run formal verification on claims",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {"200": _ok_response("Verification result")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/debates/capability-probe": {
        "post": {
            "tags": ["Auditing"],
            "summary": "Run capability probe",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {"200": _ok_response("Probe results")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/debates/deep-audit": {
        "post": {
            "tags": ["Auditing"],
            "summary": "Deep audit",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {"200": _ok_response("Audit results")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/probes/capability": {
        "post": {
            "tags": ["Auditing"],
            "summary": "Capability probe",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {"200": _ok_response("Probe results")},
            "security": [{"bearerAuth": []}],
        },
    },
}

# =============================================================================
# Document & Media Endpoints
# =============================================================================
DOCUMENT_ENDPOINTS = {
    "/api/documents": {
        "get": {
            "tags": ["Documents"],
            "summary": "List documents",
            "responses": {"200": _ok_response("Document list")},
        },
    },
    "/api/documents/formats": {
        "get": {
            "tags": ["Documents"],
            "summary": "Supported formats",
            "responses": {"200": _ok_response("Supported formats")},
        },
    },
    "/api/documents/upload": {
        "post": {
            "tags": ["Documents"],
            "summary": "Upload document",
            "requestBody": {"content": {"multipart/form-data": {"schema": {"type": "object"}}}},
            "responses": {"201": _ok_response("Document uploaded")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/podcast/feed.xml": {
        "get": {
            "tags": ["Media"],
            "summary": "Podcast RSS feed",
            "responses": {"200": {"description": "RSS feed", "content": {"application/xml": {}}}},
        },
    },
    "/api/podcast/episodes": {
        "get": {
            "tags": ["Media"],
            "summary": "Podcast episodes",
            "responses": {"200": _ok_response("Episode list")},
        },
    },
    "/api/youtube/auth": {
        "get": {
            "tags": ["Social"],
            "summary": "YouTube auth URL",
            "responses": {"200": _ok_response("Auth URL")},
        },
    },
    "/api/youtube/callback": {
        "get": {
            "tags": ["Social"],
            "summary": "YouTube OAuth callback",
            "responses": {"200": _ok_response("Auth complete")},
        },
    },
    "/api/youtube/status": {
        "get": {
            "tags": ["Social"],
            "summary": "YouTube auth status",
            "responses": {"200": _ok_response("Auth status")},
        },
    },
}

# =============================================================================
# Plugin & Laboratory Endpoints
# =============================================================================
PLUGIN_ENDPOINTS = {
    "/api/plugins": {
        "get": {
            "tags": ["Plugins"],
            "summary": "List plugins",
            "responses": {"200": _ok_response("Plugin list")},
        },
    },
    "/api/plugins/{name}": {
        "get": {
            "tags": ["Plugins"],
            "summary": "Get plugin details",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Plugin details")},
        },
    },
    "/api/plugins/{name}/run": {
        "post": {
            "tags": ["Plugins"],
            "summary": "Run plugin",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {"200": _ok_response("Plugin result")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/laboratory/emergent-traits": {
        "get": {
            "tags": ["Laboratory"],
            "summary": "Emergent traits",
            "parameters": [
                {
                    "name": "min_confidence",
                    "in": "query",
                    "schema": {"type": "number", "default": 0.5},
                },
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}},
            ],
            "responses": {"200": _ok_response("Emergent traits")},
        },
    },
    "/api/laboratory/cross-pollinations/suggest": {
        "get": {
            "tags": ["Laboratory"],
            "summary": "Cross-pollination suggestions",
            "responses": {"200": _ok_response("Suggestions")},
        },
    },
}

# =============================================================================
# Additional Endpoints
# =============================================================================
ADDITIONAL_ENDPOINTS = {
    "/api/tournaments": {
        "get": {
            "tags": ["Tournaments"],
            "summary": "List tournaments",
            "responses": {"200": _ok_response("Tournament list")},
        },
    },
    "/api/tournaments/{id}/standings": {
        "get": {
            "tags": ["Tournaments"],
            "summary": "Tournament standings",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Standings")},
        },
    },
    "/api/genesis/stats": {
        "get": {
            "tags": ["Genesis"],
            "summary": "Genesis statistics",
            "responses": {"200": _ok_response("Genesis stats")},
        },
    },
    "/api/genesis/events": {
        "get": {
            "tags": ["Genesis"],
            "summary": "Genesis events",
            "responses": {"200": _ok_response("Genesis events")},
        },
    },
    "/api/genesis/lineage/{agent}": {
        "get": {
            "tags": ["Genesis"],
            "summary": "Agent lineage",
            "parameters": [
                {"name": "agent", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Lineage data")},
        },
    },
    "/api/genesis/tree/{agent}": {
        "get": {
            "tags": ["Genesis"],
            "summary": "Agent tree",
            "parameters": [
                {"name": "agent", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Tree data")},
        },
    },
    "/api/evolution/{agent}/history": {
        "get": {
            "tags": ["Evolution"],
            "summary": "Agent evolution history",
            "parameters": [
                {"name": "agent", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Evolution history")},
        },
    },
    "/api/replays": {
        "get": {
            "tags": ["Replays"],
            "summary": "List replays",
            "responses": {"200": _ok_response("Replay list")},
        },
    },
    "/api/replays/{id}": {
        "get": {
            "tags": ["Replays"],
            "summary": "Get replay",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Replay data")},
        },
    },
    "/api/learning/evolution": {
        "get": {
            "tags": ["Learning"],
            "summary": "Learning evolution",
            "responses": {"200": _ok_response("Evolution data")},
        },
    },
    "/api/meta-learning/stats": {
        "get": {
            "tags": ["Learning"],
            "summary": "Meta-learning statistics",
            "responses": {"200": _ok_response("Meta-learning stats")},
        },
    },
    "/api/critiques/patterns": {
        "get": {
            "tags": ["Critiques"],
            "summary": "Critique patterns",
            "responses": {"200": _ok_response("Patterns")},
        },
    },
    "/api/critiques/archive": {
        "get": {
            "tags": ["Critiques"],
            "summary": "Critique archive",
            "responses": {"200": _ok_response("Archive")},
        },
    },
    "/api/reputation/all": {
        "get": {
            "tags": ["Critiques"],
            "summary": "All reputations",
            "responses": {"200": _ok_response("Reputations")},
        },
    },
    "/api/routing/best-teams": {
        "get": {
            "tags": ["Routing"],
            "summary": "Best team combinations",
            "parameters": [
                {"name": "min_debates", "in": "query", "schema": {"type": "integer", "default": 3}},
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 10}},
            ],
            "responses": {"200": _ok_response("Best teams")},
        },
    },
    "/api/routing/recommendations": {
        "post": {
            "tags": ["Routing"],
            "summary": "Agent recommendations",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "primary_domain": {"type": "string"},
                                "secondary_domains": {"type": "array", "items": {"type": "string"}},
                                "required_traits": {"type": "array", "items": {"type": "string"}},
                            },
                        }
                    }
                }
            },
            "responses": {"200": _ok_response("Recommendations")},
        },
    },
    "/api/introspection/all": {
        "get": {
            "tags": ["Introspection"],
            "summary": "All introspection data",
            "responses": {"200": _ok_response("Introspection data")},
        },
    },
    "/api/introspection/leaderboard": {
        "get": {
            "tags": ["Introspection"],
            "summary": "Introspection leaderboard",
            "responses": {"200": _ok_response("Leaderboard")},
        },
    },
    "/api/introspection/agents": {
        "get": {
            "tags": ["Introspection"],
            "summary": "Agent introspection list",
            "responses": {"200": _ok_response("Agent list")},
        },
    },
    "/api/introspection/agents/{name}": {
        "get": {
            "tags": ["Introspection"],
            "summary": "Agent introspection",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Agent introspection")},
        },
    },
}

# =============================================================================
# Combine All Endpoints
# =============================================================================
ALL_ENDPOINTS = {
    **SYSTEM_ENDPOINTS,
    **AGENT_ENDPOINTS,
    **DEBATE_ENDPOINTS,
    **ANALYTICS_ENDPOINTS,
    **CONSENSUS_ENDPOINTS,
    **RELATIONSHIP_ENDPOINTS,
    **MEMORY_ENDPOINTS,
    **BELIEF_ENDPOINTS,
    **PULSE_ENDPOINTS,
    **METRICS_ENDPOINTS,
    **VERIFICATION_ENDPOINTS,
    **DOCUMENT_ENDPOINTS,
    **PLUGIN_ENDPOINTS,
    **ADDITIONAL_ENDPOINTS,
}


def generate_openapi_schema() -> dict[str, Any]:
    """Generate complete OpenAPI 3.0 schema."""
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Aragora API",
            "description": "AI red team / decision stress-test API for Aragora LiveWire. "
            "Provides endpoints for gauntlet runs, debate-backed decision receipts, "
            "agent rankings, consensus tracking, and real-time collaboration.",
            "version": API_VERSION,
            "contact": {"name": "Aragora Team"},
            "license": {"name": "MIT"},
        },
        "servers": [
            {"url": "http://localhost:8080", "description": "Development server"},
            {"url": "https://api.aragora.ai", "description": "Production server"},
        ],
        "tags": [
            {"name": "System", "description": "Health checks and system status"},
            {"name": "Agents", "description": "Agent management, profiles, and rankings"},
            {"name": "Debates", "description": "Debate operations, history, and export"},
            {"name": "Analytics", "description": "Analysis and aggregated statistics"},
            {"name": "Insights", "description": "Position flips, moments, and patterns"},
            {"name": "Consensus", "description": "Consensus memory and settled questions"},
            {"name": "Relationships", "description": "Agent relationship tracking"},
            {"name": "Memory", "description": "Continuum memory management"},
            {"name": "Belief", "description": "Belief networks and claim analysis"},
            {"name": "Pulse", "description": "Trending topics and suggestions"},
            {"name": "Monitoring", "description": "Metrics and observability"},
            {"name": "Verification", "description": "Formal verification and proofs"},
            {"name": "Auditing", "description": "Capability probes and red teaming"},
            {"name": "Documents", "description": "Document upload and export"},
            {"name": "Media", "description": "Audio/video and podcast"},
            {"name": "Social", "description": "Social media publishing"},
            {"name": "Plugins", "description": "Plugin management and execution"},
            {"name": "Laboratory", "description": "Emergent trait analysis"},
            {"name": "Tournaments", "description": "Tournament management"},
            {"name": "Genesis", "description": "Agent genesis and lineage"},
            {"name": "Evolution", "description": "Agent evolution tracking"},
            {"name": "Replays", "description": "Debate replay management"},
            {"name": "Learning", "description": "Meta-learning statistics"},
            {"name": "Critiques", "description": "Critique patterns and reputation"},
            {"name": "Routing", "description": "Agent selection and team routing"},
            {"name": "Introspection", "description": "Agent self-awareness queries"},
        ],
        "paths": ALL_ENDPOINTS,
        "components": {
            "schemas": COMMON_SCHEMAS,
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "description": "API token authentication. Set via ARAGORA_API_TOKEN environment variable.",
                },
            },
        },
        "security": [],  # Global security is optional, per-endpoint security defined above
    }


def get_openapi_json() -> str:
    """Get OpenAPI schema as JSON string."""
    return json.dumps(generate_openapi_schema(), indent=2)


def get_openapi_yaml() -> str:
    """Get OpenAPI schema as YAML string."""
    try:
        import yaml

        return yaml.dump(generate_openapi_schema(), default_flow_style=False, sort_keys=False)
    except ImportError:
        # Fallback to JSON if PyYAML not installed
        return get_openapi_json()


def handle_openapi_request(format: str = "json") -> tuple[str, str]:
    """Handle request for OpenAPI spec.

    Returns:
        Tuple of (content, content_type)
    """
    if format == "yaml":
        return get_openapi_yaml(), "application/yaml"
    return get_openapi_json(), "application/json"


def save_openapi_schema(output_path: str = "docs/api/openapi.json") -> tuple[str, int]:
    """Save complete OpenAPI schema to file.

    Returns:
        Tuple of (file_path, endpoint_count)
    """
    schema = generate_openapi_schema()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        json.dump(schema, f, indent=2)

    endpoint_count = sum(len(methods) for methods in schema["paths"].values())
    return str(output.absolute()), endpoint_count


def get_endpoint_count() -> int:
    """Get total number of documented endpoints."""
    schema = generate_openapi_schema()
    return sum(len(methods) for methods in schema["paths"].values())


# =============================================================================
# Postman Collection Export
# =============================================================================


def generate_postman_collection(base_url: str = "{{base_url}}") -> dict[str, Any]:
    """Generate Postman Collection v2.1 from OpenAPI schema.

    Args:
        base_url: Base URL variable for requests (default: {{base_url}})

    Returns:
        Postman Collection v2.1 format dictionary
    """
    schema = generate_openapi_schema()

    # Group endpoints by tag
    tag_items: dict[str, list[dict]] = {}
    for tag in schema.get("tags", []):
        tag_items[tag["name"]] = []

    # Convert each endpoint to Postman request
    for path, methods in schema.get("paths", {}).items():
        for method, details in methods.items():
            if method in ("parameters", "servers"):
                continue

            tags = details.get("tags", ["Other"])
            tag = tags[0] if tags else "Other"

            # Build request
            request_item = _openapi_to_postman_request(
                path=path,
                method=method.upper(),
                details=details,
                base_url=base_url,
            )

            if tag not in tag_items:
                tag_items[tag] = []
            tag_items[tag].append(request_item)

    # Build folder structure
    folders = []
    for tag_name, items in tag_items.items():
        if items:
            tag_info = next(
                (t for t in schema.get("tags", []) if t["name"] == tag_name),
                {"name": tag_name, "description": ""},
            )
            folders.append(
                {
                    "name": tag_name,
                    "description": tag_info.get("description", ""),
                    "item": items,
                }
            )

    return {
        "info": {
            "_postman_id": "aragora-api-collection",
            "name": schema["info"]["title"],
            "description": schema["info"]["description"],
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
            "version": schema["info"]["version"],
        },
        "variable": [
            {
                "key": "base_url",
                "value": "http://localhost:8080",
                "type": "string",
                "description": "API base URL",
            },
            {
                "key": "api_token",
                "value": "",
                "type": "string",
                "description": "API authentication token",
            },
        ],
        "auth": {
            "type": "bearer",
            "bearer": [{"key": "token", "value": "{{api_token}}", "type": "string"}],
        },
        "item": folders,
    }


def _openapi_to_postman_request(
    path: str,
    method: str,
    details: dict[str, Any],
    base_url: str,
) -> dict[str, Any]:
    """Convert OpenAPI endpoint to Postman request format."""
    # Convert path parameters: /api/debates/{id} -> /api/debates/:id
    postman_path = path.replace("{", ":").replace("}", "")

    # Build URL parts
    url_parts = postman_path.strip("/").split("/")

    # Extract path variables
    path_variables = []
    for part in url_parts:
        if part.startswith(":"):
            var_name = part[1:]
            path_variables.append(
                {
                    "key": var_name,
                    "value": "",
                    "description": f"Path parameter: {var_name}",
                }
            )

    # Extract query parameters
    query_params = []
    for param in details.get("parameters", []):
        if param.get("in") == "query":
            query_params.append(
                {
                    "key": param["name"],
                    "value": "",
                    "description": param.get("description", ""),
                    "disabled": not param.get("required", False),
                }
            )

    # Build request body if present
    body = None
    request_body = details.get("requestBody", {})
    if request_body:
        content = request_body.get("content", {})
        if "application/json" in content:
            body = {
                "mode": "raw",
                "raw": "{}",
                "options": {"raw": {"language": "json"}},
            }

    request = {
        "name": details.get("summary", details.get("operationId", f"{method} {path}")),
        "request": {
            "method": method,
            "header": [
                {"key": "Content-Type", "value": "application/json"},
                {"key": "Accept", "value": "application/json"},
            ],
            "url": {
                "raw": f"{base_url}{postman_path}",
                "host": [base_url],
                "path": url_parts,
            },
            "description": details.get("description", ""),
        },
        "response": [],
    }

    if path_variables:
        request["request"]["url"]["variable"] = path_variables

    if query_params:
        request["request"]["url"]["query"] = query_params

    if body:
        request["request"]["body"] = body

    return request


def get_postman_json() -> str:
    """Get Postman collection as JSON string."""
    return json.dumps(generate_postman_collection(), indent=2)


def save_postman_collection(
    output_path: str = "docs/api/aragora.postman_collection.json",
) -> tuple[str, int]:
    """Save Postman collection to file.

    Returns:
        Tuple of (file_path, request_count)
    """
    collection = generate_postman_collection()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        json.dump(collection, f, indent=2)

    # Count requests
    request_count = sum(len(folder.get("item", [])) for folder in collection.get("item", []))
    return str(output.absolute()), request_count


def handle_postman_request() -> tuple[str, str]:
    """Handle request for Postman collection.

    Returns:
        Tuple of (content, content_type)
    """
    return get_postman_json(), "application/json"
