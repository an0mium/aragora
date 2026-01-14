"""System endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

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
