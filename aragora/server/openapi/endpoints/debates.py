"""Debate endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, _array_response, STANDARD_ERRORS

DEBATE_ENDPOINTS = {
    "/api/debates": {
        "get": {
            "tags": ["Debates"],
            "summary": "List debates",
            "description": """Get list of all debates for the authenticated user.

**Rate Limit:** 60 requests per minute (free), 300/min (pro), 1000/min (enterprise)

**Pagination:** Use `limit` and `offset` for pagination. Maximum 100 items per request.

**Filtering:** Results are filtered to debates owned by or shared with the authenticated user.""",
            "operationId": "listDebates",
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of debates to return",
                    "schema": {"type": "integer", "default": 20, "maximum": 100, "minimum": 1},
                    "example": 20,
                },
                {
                    "name": "offset",
                    "in": "query",
                    "description": "Number of debates to skip",
                    "schema": {"type": "integer", "default": 0, "minimum": 0},
                    "example": 0,
                },
                {
                    "name": "status",
                    "in": "query",
                    "description": "Filter by debate status",
                    "schema": {
                        "type": "string",
                        "enum": ["running", "completed", "failed", "paused"],
                    },
                },
                {
                    "name": "since",
                    "in": "query",
                    "description": "Filter debates created after this timestamp",
                    "schema": {"type": "string", "format": "date-time"},
                },
            ],
            "responses": {
                "200": _array_response("List of debates", "Debate"),
                "401": STANDARD_ERRORS["401"],
                "429": STANDARD_ERRORS["429"],
            },
            "security": [{"bearerAuth": []}],
        },
        "post": {
            "tags": ["Debates"],
            "summary": "Create a new debate",
            "description": """Start a new multi-agent debate on a given topic.

**Rate Limit:** 10 debates per day (free), 100/day (pro), unlimited (enterprise)

**Concurrent Limit:** 1 concurrent debate (free), 5 (pro), 20 (enterprise)

**Authentication:** Required. The debate will be owned by the authenticated user.

**WebSocket:** After creation, connect to the returned `websocket_url` to stream real-time debate progress.""",
            "operationId": "createDebate",
            "security": [{"bearerAuth": []}],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/DebateCreateRequest"},
                        "examples": {
                            "simple": {
                                "summary": "Simple debate",
                                "value": {
                                    "task": "Should we use TypeScript for our next project?",
                                    "rounds": 3,
                                },
                            },
                            "with_agents": {
                                "summary": "Debate with specific agents",
                                "value": {
                                    "task": "Is GraphQL better than REST for mobile apps?",
                                    "agents": ["claude", "gpt-4", "gemini"],
                                    "rounds": 3,
                                    "consensus": "majority",
                                },
                            },
                            "with_context": {
                                "summary": "Debate with context",
                                "value": {
                                    "task": "Should we migrate to Kubernetes?",
                                    "context": "We have 50 microservices and 10 engineers.",
                                    "rounds": 5,
                                    "consensus": "weighted",
                                },
                            },
                        },
                    }
                },
            },
            "responses": {
                "200": _ok_response("Debate created successfully", "DebateCreateResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "402": STANDARD_ERRORS["402"],
                "429": STANDARD_ERRORS["429"],
            },
        },
    },
    "/api/debate": {
        "post": {
            "tags": ["Debates"],
            "summary": "Create a new debate (deprecated)",
            "description": "Deprecated. Use POST /api/debates instead.",
            "operationId": "createDebateDeprecated",
            "deprecated": True,
            "security": [{"bearerAuth": []}],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/DebateCreateRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response("Debate created successfully", "DebateCreateResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
            },
        },
    },
    "/api/debates/{id}": {
        "get": {
            "tags": ["Debates"],
            "summary": "Get debate by ID",
            "operationId": "getDebateById",
            "description": "Retrieve full details for a specific debate by its unique identifier.",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Debate details", "Debate"),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/debates/slug/{slug}": {
        "get": {
            "tags": ["Debates"],
            "summary": "Get debate by slug",
            "operationId": "getDebateBySlug",
            "description": "Retrieve full details for a specific debate by its URL-friendly slug.",
            "parameters": [
                {"name": "slug", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Debate details", "Debate"),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/debates/{id}/messages": {
        "get": {
            "tags": ["Debates"],
            "summary": "Get debate messages",
            "operationId": "getDebateMessages",
            "description": "Get paginated message history for a debate",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 50, "maximum": 200},
                },
                {"name": "offset", "in": "query", "schema": {"type": "integer", "default": 0}},
            ],
            "responses": {"200": _ok_response("Paginated messages")},
        },
    },
    "/api/debates/{id}/convergence": {
        "get": {
            "tags": ["Debates"],
            "summary": "Get convergence status",
            "operationId": "getDebateConvergence",
            "description": "Check if debate has reached semantic convergence",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Convergence status")},
        },
    },
    "/api/debates/{id}/citations": {
        "get": {
            "tags": ["Debates"],
            "summary": "Get evidence citations",
            "operationId": "getDebateCitations",
            "description": "Get grounded verdict with evidence citations",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Citations and grounding score")},
        },
    },
    "/api/debates/{id}/evidence": {
        "get": {
            "tags": ["Debates"],
            "summary": "Get debate evidence",
            "operationId": "getDebateEvidence",
            "description": "Get all evidence and sources cited during the debate.",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Evidence data")},
        },
    },
    "/api/debates/{id}/impasse": {
        "get": {
            "tags": ["Debates"],
            "summary": "Get impasse status",
            "operationId": "getDebateImpasse",
            "description": "Check if debate reached an impasse",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Impasse status")},
        },
    },
    "/api/debates/{id}/meta-critique": {
        "get": {
            "tags": ["Debates"],
            "summary": "Get meta-critique",
            "operationId": "getDebateMetaCritique",
            "description": "Get meta-level critique analyzing the debate's reasoning quality.",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Meta-critique data")},
        },
    },
    "/api/debates/{id}/graph/stats": {
        "get": {
            "tags": ["Debates"],
            "summary": "Get debate graph stats",
            "operationId": "getDebateGraphStats",
            "description": "Get graph statistics showing argument structure and relationships.",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Graph statistics")},
        },
    },
    "/api/debates/{id}/fork": {
        "post": {
            "tags": ["Debates"],
            "summary": "Fork debate",
            "operationId": "forkDebate",
            "description": "Create a counterfactual branch from a specific round",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "branch_point": {
                                    "type": "integer",
                                    "description": "Round to branch from",
                                },
                                "new_premise": {
                                    "type": "string",
                                    "description": "New premise for fork",
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "201": _ok_response("Forked debate created"),
                "400": STANDARD_ERRORS["400"],
            },
        },
    },
    "/api/debates/{id}/export/{format}": {
        "get": {
            "tags": ["Debates"],
            "summary": "Export debate",
            "operationId": "exportDebate",
            "description": "Export debate content in the specified format for sharing or archiving.",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {
                    "name": "format",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string", "enum": ["json", "markdown", "html", "pdf"]},
                },
            ],
            "responses": {"200": _ok_response("Exported debate")},
        },
    },
    "/api/debates/{id}/broadcast": {
        "post": {
            "tags": ["Debates"],
            "summary": "Generate debate broadcast",
            "operationId": "createDebateBroadcast",
            "description": "Generate audio/video broadcast of debate",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "format": {"type": "string", "enum": ["audio", "video"]},
                                "voices": {"type": "object"},
                            },
                        }
                    }
                },
            },
            "responses": {"202": _ok_response("Broadcast generation started")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/debates/{id}/publish/twitter": {
        "post": {
            "tags": ["Social"],
            "summary": "Publish to Twitter",
            "operationId": "publishDebateToTwitter",
            "description": "Publish debate summary and key points to Twitter/X.",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Published to Twitter")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/debates/{id}/publish/youtube": {
        "post": {
            "tags": ["Social"],
            "summary": "Publish to YouTube",
            "operationId": "publishDebateToYouTube",
            "description": "Publish debate video/audio to YouTube.",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Published to YouTube")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/debates/{id}/red-team": {
        "get": {
            "tags": ["Auditing"],
            "summary": "Get red team results",
            "operationId": "getDebateRedTeamResults",
            "description": "Get adversarial analysis results from red team review.",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Red team results")},
        },
    },
    "/api/search": {
        "get": {
            "tags": ["Debates"],
            "summary": "Cross-debate search",
            "operationId": "searchDebates",
            "description": "Search across all debates for matching content and arguments.",
            "parameters": [
                {"name": "q", "in": "query", "required": True, "schema": {"type": "string"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}},
            ],
            "responses": {"200": _ok_response("Search results")},
        },
    },
    "/api/dashboard/debates": {
        "get": {
            "tags": ["Debates"],
            "summary": "Dashboard debates",
            "operationId": "getDashboardDebates",
            "description": "Get debates formatted for dashboard display",
            "parameters": [
                {"name": "domain", "in": "query", "schema": {"type": "string"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 10}},
                {"name": "hours", "in": "query", "schema": {"type": "integer", "default": 24}},
            ],
            "responses": {"200": _ok_response("Dashboard data")},
        },
    },
}
