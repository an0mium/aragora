"""Control plane endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS, AUTH_REQUIREMENTS


CONTROL_PLANE_ENDPOINTS = {
    "/api/control-plane/agents": {
        "get": {
            "tags": ["Control Plane"],
            "summary": "List control plane agents",
            "description": "List registered agents. Supports filtering by capability.",
            "parameters": [
                {
                    "name": "capability",
                    "in": "query",
                    "description": "Filter agents by capability",
                    "schema": {"type": "string"},
                },
                {
                    "name": "available",
                    "in": "query",
                    "description": "Only show available agents (default: true)",
                    "schema": {"type": "boolean", "default": True},
                },
            ],
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "responses": {
                "200": _ok_response("Agent list", "ControlPlaneAgentList"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "post": {
            "tags": ["Control Plane"],
            "summary": "Register agent",
            "description": "Register an agent with capabilities and model metadata.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["agent_id", "capabilities"],
                            "properties": {
                                "agent_id": {"type": "string"},
                                "capabilities": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "model": {"type": "string"},
                                "provider": {"type": "string"},
                                "metadata": {"type": "object"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "201": _ok_response("Agent registered", "ControlPlaneAgent"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/control-plane/agents/{agent_id}": {
        "get": {
            "tags": ["Control Plane"],
            "summary": "Get agent",
            "description": "Get a specific agent by ID.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "agent_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Agent details", "ControlPlaneAgent"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "delete": {
            "tags": ["Control Plane"],
            "summary": "Unregister agent",
            "description": "Unregister an agent by ID.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "agent_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Agent unregistered"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/control-plane/agents/{agent_id}/heartbeat": {
        "post": {
            "tags": ["Control Plane"],
            "summary": "Send heartbeat",
            "description": "Update agent heartbeat and status.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "agent_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Heartbeat accepted"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/control-plane/tasks": {
        "post": {
            "tags": ["Control Plane"],
            "summary": "Submit task",
            "description": "Submit a task to the control plane scheduler.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["task_type"],
                            "properties": {
                                "task_type": {"type": "string"},
                                "payload": {"type": "object"},
                                "required_capabilities": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "priority": {
                                    "type": "string",
                                    "enum": ["low", "normal", "high", "urgent"],
                                },
                                "timeout_seconds": {"type": "number"},
                                "metadata": {"type": "object"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "201": _ok_response("Task submitted", "ControlPlaneTaskCreated"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/control-plane/tasks/{task_id}": {
        "get": {
            "tags": ["Control Plane"],
            "summary": "Get task",
            "description": "Get task status and metadata by ID.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "task_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Task details", "ControlPlaneTask"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/control-plane/tasks/{task_id}/complete": {
        "post": {
            "tags": ["Control Plane"],
            "summary": "Complete task",
            "description": "Mark a task as completed.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "task_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "result": {"type": "object"},
                                "agent_id": {"type": "string"},
                                "latency_ms": {"type": "number"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Task completed"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/control-plane/tasks/{task_id}/fail": {
        "post": {
            "tags": ["Control Plane"],
            "summary": "Fail task",
            "description": "Mark a task as failed.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "task_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "error": {"type": "string"},
                                "agent_id": {"type": "string"},
                                "result": {"type": "object"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Task failed"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/control-plane/tasks/{task_id}/cancel": {
        "post": {
            "tags": ["Control Plane"],
            "summary": "Cancel task",
            "description": "Cancel a task by ID.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "task_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Task cancelled"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/control-plane/tasks/claim": {
        "post": {
            "tags": ["Control Plane"],
            "summary": "Claim task",
            "description": "Claim the next available task for an agent.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["agent_id"],
                            "properties": {
                                "agent_id": {"type": "string"},
                                "capabilities": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "block_ms": {"type": "integer", "default": 5000},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Task claim result", "ControlPlaneTaskClaimResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/control-plane/queue": {
        "get": {
            "tags": ["Control Plane"],
            "summary": "Queue snapshot",
            "description": "Get pending and running tasks for dashboard queue.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 50},
                }
            ],
            "responses": {
                "200": _ok_response("Queue snapshot", "ControlPlaneQueue"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/control-plane/metrics": {
        "get": {
            "tags": ["Control Plane"],
            "summary": "Control plane metrics",
            "description": "Get dashboard metrics derived from scheduler and registry stats.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "responses": {
                "200": _ok_response("Metrics snapshot", "ControlPlaneMetrics"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/control-plane/stats": {
        "get": {
            "tags": ["Control Plane"],
            "summary": "Control plane stats",
            "description": "Get scheduler and registry stats.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "responses": {
                "200": _ok_response("Control plane stats", "ControlPlaneStats"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/control-plane/health": {
        "get": {
            "tags": ["Control Plane"],
            "summary": "System health",
            "description": "Get system health with agent health summaries.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "responses": {
                "200": _ok_response("System health", "ControlPlaneHealth"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/control-plane/health/{agent_id}": {
        "get": {
            "tags": ["Control Plane"],
            "summary": "Agent health",
            "description": "Get health status for a specific agent.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "agent_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Agent health"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/control-plane/deliberations": {
        "post": {
            "tags": ["Control Plane"],
            "summary": "Run robust decisionmaking session",
            "description": "Run or queue a robust decisionmaking session (async when async=true).",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/DeliberationRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response("Decisionmaking completed", "DeliberationSyncResponse"),
                "202": _ok_response("Decisionmaking queued", "DeliberationQueuedResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/control-plane/deliberations/{request_id}": {
        "get": {
            "tags": ["Control Plane"],
            "summary": "Get robust decisionmaking result",
            "description": "Fetch a stored robust decisionmaking record by request ID.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "request_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Decisionmaking record", "DeliberationRecord"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/control-plane/deliberations/{request_id}/status": {
        "get": {
            "tags": ["Control Plane"],
            "summary": "Get robust decisionmaking status",
            "description": "Check robust decisionmaking status for polling.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "request_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Decisionmaking status", "DeliberationStatus"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
}
