"""Debate endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, _array_response, STANDARD_ERRORS

DEBATE_ENDPOINTS = {
    "/api/debates": {
        "get": {
            "tags": ["Debates"],
            "summary": "List debates",
            "description": "Get list of all debates (requires auth)",
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 20, "maximum": 100},
                },
                {"name": "offset", "in": "query", "schema": {"type": "integer", "default": 0}},
            ],
            "responses": {"200": _array_response("List of debates", "Debate")},
            "security": [{"bearerAuth": []}],
        },
        "post": {
            "tags": ["Debates"],
            "summary": "Create a new debate",
            "description": "Start a new multi-agent debate on a given topic",
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
