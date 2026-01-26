"""Additional endpoint definitions (tournaments, genesis, evolution, etc.)."""

from aragora.server.openapi.helpers import _ok_response

ADDITIONAL_ENDPOINTS = {
    "/api/tournaments": {
        "get": {
            "tags": ["Tournaments"],
            "summary": "List tournaments",
            "description": "Get list of all agent tournaments and their status.",
            "operationId": "listTournaments",
            "responses": {"200": _ok_response("Tournament list")},
        },
    },
    "/api/tournaments/{id}/standings": {
        "get": {
            "tags": ["Tournaments"],
            "summary": "Tournament standings",
            "description": "Get current standings and rankings for a tournament.",
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
            "description": "Get statistics about agent genesis events and population.",
            "operationId": "listGenesisStats",
            "responses": {"200": _ok_response("Genesis stats")},
        },
    },
    "/api/genesis/events": {
        "get": {
            "tags": ["Genesis"],
            "summary": "Genesis events",
            "description": "Get timeline of agent creation and evolution events.",
            "operationId": "listGenesisEvents",
            "responses": {"200": _ok_response("Genesis events")},
        },
    },
    "/api/genesis/lineage/{agent}": {
        "get": {
            "tags": ["Genesis"],
            "summary": "Agent lineage",
            "description": "Get lineage tree showing agent ancestry and descendants.",
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
            "description": "Get hierarchical tree view of agent relationships.",
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
            "description": "Get history of agent evolution including version changes and improvements.",
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
            "description": "Get list of available debate replays.",
            "operationId": "listReplays",
            "responses": {"200": _ok_response("Replay list")},
        },
    },
    "/api/replays/{id}": {
        "get": {
            "tags": ["Replays"],
            "summary": "Get replay",
            "description": "Get detailed replay data for a specific debate.",
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
            "description": "Get learning evolution data showing system improvements over time.",
            "operationId": "listLearningEvolution",
            "responses": {"200": _ok_response("Evolution data")},
        },
    },
    "/api/meta-learning/stats": {
        "get": {
            "tags": ["Learning"],
            "summary": "Meta-learning statistics",
            "description": "Get statistics from meta-learning processes.",
            "operationId": "listMetaLearningStats",
            "responses": {"200": _ok_response("Meta-learning stats")},
        },
    },
    "/api/critiques/patterns": {
        "get": {
            "tags": ["Critiques"],
            "summary": "Critique patterns",
            "description": "Get common critique patterns across debates.",
            "operationId": "listCritiquesPatterns",
            "responses": {"200": _ok_response("Patterns")},
        },
    },
    "/api/critiques/archive": {
        "get": {
            "tags": ["Critiques"],
            "summary": "Critique archive",
            "description": "Get archived critiques for historical analysis.",
            "operationId": "listCritiquesArchive",
            "responses": {"200": _ok_response("Archive")},
        },
    },
    "/api/reputation/all": {
        "get": {
            "tags": ["Critiques"],
            "summary": "All reputations",
            "description": "Get reputation data for all agents.",
            "operationId": "listReputationAll",
            "responses": {"200": _ok_response("Reputations")},
        },
    },
    "/api/routing/best-teams": {
        "get": {
            "tags": ["Routing"],
            "summary": "Best team combinations",
            "description": "Get recommended team combinations based on historical performance.",
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
            "description": "Get agent recommendations based on domain requirements and traits.",
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
            "description": "Get comprehensive introspection data across all agents.",
            "operationId": "listIntrospectionAll",
            "responses": {"200": _ok_response("Introspection data")},
        },
    },
    "/api/introspection/leaderboard": {
        "get": {
            "tags": ["Introspection"],
            "summary": "Introspection leaderboard",
            "description": "Get leaderboard rankings based on introspection metrics.",
            "operationId": "listIntrospectionLeaderboard",
            "responses": {"200": _ok_response("Leaderboard")},
        },
    },
    "/api/introspection/agents": {
        "get": {
            "tags": ["Introspection"],
            "summary": "Agent introspection list",
            "description": "Get list of agents with introspection data available.",
            "operationId": "listIntrospectionAgents",
            "responses": {"200": _ok_response("Agent list")},
        },
    },
    "/api/introspection/agents/{name}": {
        "get": {
            "tags": ["Introspection"],
            "summary": "Agent introspection",
            "description": "Get detailed introspection data for a specific agent.",
            "operationId": "getIntrospectionAgent",
            "parameters": [
                {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Agent introspection")},
        },
    },
}
