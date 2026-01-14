"""Belief network endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

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
