"""
Debate API Hardening Endpoints.

Additional debate endpoints for participation, intervention, and cost estimation
that complete the Debate-as-a-Service API surface.
"""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS

DEBATE_HARDENING_ENDPOINTS = {
    # =========================================================================
    # Debate Participation Endpoints (join, vote, suggest, update)
    # =========================================================================
    "/api/v1/debates/{id}/join": {
        "post": {
            "tags": ["Debates"],
            "summary": "Join debate",
            "operationId": "joinDebateV1",
            "description": "Join an active debate as an observer, participant, or moderator.",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/DebateJoinRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response(
                    "Joined debate",
                    {
                        "success": {"type": "boolean"},
                        "participant_id": {"type": "string"},
                        "role": {"type": "string"},
                        "websocket_url": {"type": "string"},
                    },
                ),
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/debates/{id}/vote": {
        "post": {
            "tags": ["Debates"],
            "summary": "Submit vote",
            "operationId": "submitDebateVoteV1",
            "description": "Submit a vote on the current debate position. Votes influence consensus detection.",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/DebateVoteRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response(
                    "Vote recorded",
                    {
                        "success": {"type": "boolean"},
                        "vote_id": {"type": "string"},
                        "position": {"type": "string"},
                        "intensity": {"type": "integer"},
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/debates/{id}/suggest": {
        "post": {
            "tags": ["Debates"],
            "summary": "Submit suggestion",
            "operationId": "submitDebateSuggestionV1",
            "description": "Submit a suggestion or argument to be considered by debate agents in the next round.",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/DebateSuggestionRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response(
                    "Suggestion recorded",
                    {
                        "success": {"type": "boolean"},
                        "suggestion_id": {"type": "string"},
                        "will_appear_in": {"type": "string"},
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/debates/{id}/update": {
        "put": {
            "tags": ["Debates"],
            "summary": "Update debate configuration",
            "operationId": "updateDebateV1",
            "description": "Update a debate's configuration. Can modify rounds, consensus strategy, and context.",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/DebateUpdateRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response(
                    "Debate updated",
                    {
                        "success": {"type": "boolean"},
                        "debate_id": {"type": "string"},
                        "updated_fields": {"type": "array", "items": {"type": "string"}},
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    # =========================================================================
    # Intervention Endpoints
    # =========================================================================
    "/api/debates/{debate_id}/intervention/pause": {
        "post": {
            "tags": ["Debates", "Interventions"],
            "summary": "Pause debate",
            "operationId": "pauseDebateIntervention",
            "description": "Pause an active debate. Stops agent responses but preserves state.",
            "parameters": [
                {"name": "debate_id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response(
                    "Debate paused",
                    {
                        "success": {"type": "boolean"},
                        "debate_id": {"type": "string"},
                        "is_paused": {"type": "boolean"},
                        "paused_at": {"type": "string", "format": "date-time"},
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/debates/{debate_id}/intervention/resume": {
        "post": {
            "tags": ["Debates", "Interventions"],
            "summary": "Resume debate",
            "operationId": "resumeDebateIntervention",
            "description": "Resume a paused debate from where it left off.",
            "parameters": [
                {"name": "debate_id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response(
                    "Debate resumed",
                    {
                        "success": {"type": "boolean"},
                        "debate_id": {"type": "string"},
                        "is_paused": {"type": "boolean"},
                        "resumed_at": {"type": "string", "format": "date-time"},
                        "pause_duration_seconds": {"type": "number", "nullable": True},
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/debates/{debate_id}/intervention/inject": {
        "post": {
            "tags": ["Debates", "Interventions"],
            "summary": "Inject argument",
            "operationId": "injectDebateArgument",
            "description": "Inject a user argument or follow-up into the debate. Included in next round's context.",
            "parameters": [
                {"name": "debate_id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/DebateInjectArgumentRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response(
                    "Argument injected",
                    {
                        "success": {"type": "boolean"},
                        "debate_id": {"type": "string"},
                        "injection_id": {"type": "string"},
                        "type": {"type": "string"},
                        "will_appear_in": {"type": "string"},
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/debates/{debate_id}/intervention/weights": {
        "post": {
            "tags": ["Debates", "Interventions"],
            "summary": "Update agent weight",
            "operationId": "updateDebateAgentWeight",
            "description": "Update an agent's influence weight. Affects consensus voting.",
            "parameters": [
                {"name": "debate_id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/DebateUpdateWeightsRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response(
                    "Weight updated",
                    {
                        "success": {"type": "boolean"},
                        "debate_id": {"type": "string"},
                        "agent": {"type": "string"},
                        "old_weight": {"type": "number"},
                        "new_weight": {"type": "number"},
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/debates/{debate_id}/intervention/threshold": {
        "post": {
            "tags": ["Debates", "Interventions"],
            "summary": "Update consensus threshold",
            "operationId": "updateDebateThreshold",
            "description": "Update the consensus threshold (0.5=majority, 0.75=strong, 1.0=unanimous).",
            "parameters": [
                {"name": "debate_id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/DebateUpdateThresholdRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response(
                    "Threshold updated",
                    {
                        "success": {"type": "boolean"},
                        "debate_id": {"type": "string"},
                        "old_threshold": {"type": "number"},
                        "new_threshold": {"type": "number"},
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/debates/{debate_id}/intervention/state": {
        "get": {
            "tags": ["Debates", "Interventions"],
            "summary": "Get intervention state",
            "operationId": "getDebateInterventionState",
            "description": "Get current intervention state: pause status, weights, pending injections.",
            "parameters": [
                {"name": "debate_id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response(
                    "Intervention state",
                    {
                        "debate_id": {"type": "string"},
                        "is_paused": {"type": "boolean"},
                        "paused_at": {"type": "string", "format": "date-time", "nullable": True},
                        "consensus_threshold": {"type": "number"},
                        "agent_weights": {"type": "object"},
                        "pending_injections": {"type": "integer"},
                        "pending_follow_ups": {"type": "integer"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/debates/{debate_id}/intervention/log": {
        "get": {
            "tags": ["Debates", "Interventions"],
            "summary": "Get intervention log",
            "operationId": "getDebateInterventionLog",
            "description": "Get audit log of all interventions for a debate.",
            "parameters": [
                {"name": "debate_id", "in": "path", "required": True, "schema": {"type": "string"}},
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 50, "maximum": 200},
                },
            ],
            "responses": {
                "200": _ok_response(
                    "Intervention log",
                    {
                        "debate_id": {"type": "string"},
                        "total_interventions": {"type": "integer"},
                        "interventions": {"type": "array", "items": {"type": "object"}},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    # =========================================================================
    # Cost Estimation Endpoint
    # =========================================================================
    "/api/v1/debates/estimate-cost": {
        "get": {
            "tags": ["Debates", "Billing"],
            "summary": "Estimate debate cost",
            "operationId": "estimateDebateCost",
            "description": "Estimate cost of running a debate before creating it. Returns per-model breakdown.",
            "parameters": [
                {
                    "name": "num_agents",
                    "in": "query",
                    "description": "Number of agents",
                    "schema": {"type": "integer", "default": 3, "minimum": 1, "maximum": 8},
                },
                {
                    "name": "num_rounds",
                    "in": "query",
                    "description": "Number of debate rounds",
                    "schema": {"type": "integer", "default": 9, "minimum": 1, "maximum": 12},
                },
                {
                    "name": "model_types",
                    "in": "query",
                    "description": "Comma-separated model types",
                    "schema": {"type": "string"},
                    "example": "claude-sonnet-4,gpt-4o,gemini-pro",
                },
            ],
            "responses": {
                "200": _ok_response(
                    "Cost estimate",
                    "DebateCostEstimateResponse",
                ),
                "400": STANDARD_ERRORS["400"],
            },
        },
    },
}
