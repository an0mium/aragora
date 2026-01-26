"""Additional endpoint definitions (tournaments, genesis, evolution, etc.)."""

from aragora.server.openapi.helpers import _ok_response

ADDITIONAL_ENDPOINTS = {
    "/api/tournaments": {
        "get": {
            "tags": ["Tournaments"],
            "summary": "List tournaments",
            "operationId": "listTournaments",
            "responses": {"200": _ok_response("Tournament list")},
        },
    },
    "/api/tournaments/{id}/standings": {
        "get": {
            "tags": ["Tournaments"],
            "summary": "Tournament standings",
            "operationId": "getTournamentsStanding",
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
            "operationId": "listGenesisStats",
            "responses": {"200": _ok_response("Genesis stats")},
        },
    },
    "/api/genesis/events": {
        "get": {
            "tags": ["Genesis"],
            "summary": "Genesis events",
            "operationId": "listGenesisEvents",
            "responses": {"200": _ok_response("Genesis events")},
        },
    },
    "/api/genesis/lineage/{agent}": {
        "get": {
            "tags": ["Genesis"],
            "summary": "Agent lineage",
            "operationId": "getGenesisLineage",
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
            "operationId": "getGenesisTree",
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
            "operationId": "getEvolutionHistory",
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
            "operationId": "listReplays",
            "responses": {"200": _ok_response("Replay list")},
        },
    },
    "/api/replays/{id}": {
        "get": {
            "tags": ["Replays"],
            "summary": "Get replay",
            "operationId": "getReplay",
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
            "operationId": "listLearningEvolution",
            "responses": {"200": _ok_response("Evolution data")},
        },
    },
    "/api/meta-learning/stats": {
        "get": {
            "tags": ["Learning"],
            "summary": "Meta-learning statistics",
            "operationId": "listMetaLearningStats",
            "responses": {"200": _ok_response("Meta-learning stats")},
        },
    },
    "/api/critiques/patterns": {
        "get": {
            "tags": ["Critiques"],
            "summary": "Critique patterns",
            "operationId": "listCritiquesPatterns",
            "responses": {"200": _ok_response("Patterns")},
        },
    },
    "/api/critiques/archive": {
        "get": {
            "tags": ["Critiques"],
            "summary": "Critique archive",
            "operationId": "listCritiquesArchive",
            "responses": {"200": _ok_response("Archive")},
        },
    },
    "/api/reputation/all": {
        "get": {
            "tags": ["Critiques"],
            "summary": "All reputations",
            "operationId": "listReputationAll",
            "responses": {"200": _ok_response("Reputations")},
        },
    },
    "/api/routing/best-teams": {
        "get": {
            "tags": ["Routing"],
            "summary": "Best team combinations",
            "operationId": "listRoutingBestTeams",
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
            "operationId": "createRoutingRecommendations",
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
            "operationId": "listIntrospectionAll",
            "responses": {"200": _ok_response("Introspection data")},
        },
    },
    "/api/introspection/leaderboard": {
        "get": {
            "tags": ["Introspection"],
            "summary": "Introspection leaderboard",
            "operationId": "listIntrospectionLeaderboard",
            "responses": {"200": _ok_response("Leaderboard")},
        },
    },
    "/api/introspection/agents": {
        "get": {
            "tags": ["Introspection"],
            "summary": "Agent introspection list",
            "operationId": "listIntrospectionAgents",
            "responses": {"200": _ok_response("Agent list")},
        },
    },
    "/api/introspection/agents/{name}": {
        "get": {
            "tags": ["Introspection"],
            "summary": "Agent introspection",
            "operationId": "getIntrospectionAgent",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Agent introspection")},
        },
    },
}
