"""OpenAPI endpoint definitions for FastAPI v2 orchestration routes."""

from aragora.server.openapi.helpers import _ok_response, AUTH_REQUIREMENTS, STANDARD_ERRORS

_REQUEST_ID_PARAM = {
    "name": "request_id",
    "in": "path",
    "required": True,
    "schema": {"type": "string"},
    "description": "Deliberation request ID.",
}

_RESULT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "request_id": {"type": "string"},
        "success": {"type": "boolean"},
        "consensus_reached": {"type": "boolean"},
        "final_answer": {"type": "string"},
        "confidence": {"type": "number"},
        "agents_participated": {"type": "array", "items": {"type": "string"}},
        "rounds_completed": {"type": "integer"},
        "duration_seconds": {"type": "number"},
        "knowledge_context_used": {"type": "array", "items": {"type": "string"}},
        "channels_notified": {"type": "array", "items": {"type": "string"}},
        "receipt_id": {"type": "string"},
        "error": {"type": "string"},
        "created_at": {"type": "string", "format": "date-time"},
        "estimated_cost_usd": {"type": "number"},
    },
}

_STATUS_RESPONSES = {
    "200": _ok_response(
        "Deliberation status.",
        {
            "request_id": {"type": "string"},
            "status": {"type": "string"},
            "result": _RESULT_RESPONSE_SCHEMA,
        },
    ),
    "401": STANDARD_ERRORS["401"],
    "403": STANDARD_ERRORS["403"],
    "404": STANDARD_ERRORS["404"],
    "500": STANDARD_ERRORS["500"],
}

_TEMPLATE_LIST_SCHEMA = {
    "templates": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "default_agents": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "default_knowledge_sources": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "output_format": {"type": "string"},
                "consensus_threshold": {"type": "number"},
                "max_rounds": {"type": "integer"},
                "personas": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    "count": {"type": "integer"},
}

_TEMPLATE_LIST_PARAMETERS = [
    {
        "name": "category",
        "in": "query",
        "description": "Filter by template category.",
        "schema": {"type": "string"},
    },
    {
        "name": "search",
        "in": "query",
        "description": "Search term for name/description.",
        "schema": {"type": "string"},
    },
    {
        "name": "tags",
        "in": "query",
        "description": "Comma-separated tags.",
        "schema": {"type": "string"},
    },
    {
        "name": "limit",
        "in": "query",
        "description": "Maximum rows to return.",
        "schema": {"type": "integer", "minimum": 1, "maximum": 500, "default": 50},
    },
    {
        "name": "offset",
        "in": "query",
        "description": "Pagination offset.",
        "schema": {"type": "integer", "minimum": 0, "default": 0},
    },
]

_TEMPLATE_LIST_RESPONSES = {
    "200": _ok_response("Orchestration template list.", _TEMPLATE_LIST_SCHEMA),
    "401": STANDARD_ERRORS["401"],
    "403": STANDARD_ERRORS["403"],
    "500": STANDARD_ERRORS["500"],
}

ORCHESTRATION_ENDPOINTS = {
    "/api/v1/orchestration/status/{request_id}": {
        "get": {
            "tags": ["Orchestration"],
            "summary": "Get deliberation status (v1 compatibility)",
            "operationId": "getOrchestrationDeliberationStatusV1Compat",
            "description": (
                "Get orchestration deliberation status for the legacy v1 compatibility path."
            ),
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [_REQUEST_ID_PARAM],
            "responses": _STATUS_RESPONSES,
        }
    },
    "/api/v1/orchestration/templates": {
        "get": {
            "tags": ["Orchestration"],
            "summary": "List orchestration templates (v1 compatibility)",
            "operationId": "listOrchestrationTemplatesV1Compat",
            "description": "List orchestration templates on the legacy v1 compatibility path.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": _TEMPLATE_LIST_PARAMETERS,
            "responses": _TEMPLATE_LIST_RESPONSES,
        }
    },
    "/api/v2/orchestration/status/{request_id}": {
        "get": {
            "tags": ["Orchestration"],
            "summary": "Get deliberation status",
            "operationId": "getOrchestrationDeliberationStatusV2",
            "description": "Get orchestration deliberation status and optional result payload.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [_REQUEST_ID_PARAM],
            "responses": _STATUS_RESPONSES,
        }
    },
}
