"""Relationship endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

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
