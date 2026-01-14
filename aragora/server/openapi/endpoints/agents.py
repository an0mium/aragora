"""Agent endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, _array_response, STANDARD_ERRORS

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
            "responses": {
                "200": _ok_response("Agent profile", "Agent"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
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
