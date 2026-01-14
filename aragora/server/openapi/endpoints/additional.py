"""Additional endpoint definitions (tournaments, genesis, evolution, etc.)."""

from aragora.server.openapi.helpers import _ok_response

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
