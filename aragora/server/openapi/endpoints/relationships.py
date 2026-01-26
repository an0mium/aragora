"""Relationship endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

RELATIONSHIP_ENDPOINTS = {
    "/api/relationships/summary": {
        "get": {
            "tags": ["Relationships"],
            "summary": "Relationship summary",
            "operationId": "listRelationshipsSummary",
            "description": "Get a summary of all agent relationships including collaboration scores and interaction history.",
            "responses": {"200": _ok_response("Relationship summary")},
        },
    },
    "/api/relationships/graph": {
        "get": {
            "tags": ["Relationships"],
            "summary": "Relationship graph",
            "operationId": "listRelationshipsGraph",
            "description": "Get graph data for agent relationships",
            "responses": {"200": _ok_response("Graph data")},
        },
    },
    "/api/relationships/stats": {
        "get": {
            "tags": ["Relationships"],
            "summary": "Relationship statistics",
            "operationId": "listRelationshipsStats",
            "description": "Get aggregate statistics about agent relationships across all debates.",
            "responses": {"200": _ok_response("Relationship stats")},
        },
    },
    "/api/relationship/{agent_a}/{agent_b}": {
        "get": {
            "tags": ["Relationships"],
            "summary": "Get relationship between two agents",
            "operationId": "getRelationship",
            "description": "Get detailed relationship data between two specific agents including interaction history.",
            "parameters": [
                {
                    "name": "agent_a",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "First agent identifier",
                },
                {
                    "name": "agent_b",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Second agent identifier",
                },
            ],
            "responses": {"200": _ok_response("Relationship data", "Relationship")},
        },
    },
}
