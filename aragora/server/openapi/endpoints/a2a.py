"""
OpenAPI endpoint definitions for A2A (Agent-to-Agent) Protocol.

The A2A protocol enables interoperability between AI agents, allowing them
to discover each other's capabilities and delegate tasks.
"""

from aragora.server.openapi.helpers import (
    _ok_response,
    STANDARD_ERRORS,
)

A2A_ENDPOINTS = {
    "/.well-known/agent.json": {
        "get": {
            "tags": ["A2A Protocol"],
            "summary": "Agent discovery",
            "description": """Discovery endpoint for A2A protocol.

Returns the agent card for the primary Aragora agent, enabling
other agents to discover capabilities and endpoints.

**A2A Specification:** This follows the Agent-to-Agent protocol standard
for agent discovery at the well-known location.

**Response includes:**
- Agent name and version
- Supported capabilities (debate, audit, critique, research)
- Endpoint URLs for tasks and agent listing""",
            "operationId": "getAgentDiscovery",
            "responses": {
                "200": {
                    "description": "Agent card",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "Agent identifier",
                                        "example": "aragora",
                                    },
                                    "version": {
                                        "type": "string",
                                        "description": "Agent version",
                                        "example": "1.0.0",
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Human-readable description",
                                    },
                                    "capabilities": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of supported capabilities",
                                        "example": ["debate", "audit", "critique", "research"],
                                    },
                                    "endpoints": {
                                        "type": "object",
                                        "properties": {
                                            "agents": {"type": "string"},
                                            "tasks": {"type": "string"},
                                        },
                                        "description": "Available API endpoints",
                                    },
                                },
                            }
                        }
                    },
                },
            },
        },
    },
    "/api/v1/a2a/.well-known/agent.json": {
        "get": {
            "tags": ["A2A Protocol"],
            "summary": "Agent discovery (API path)",
            "description": """Alternative discovery endpoint under the API path.

Same as `/.well-known/agent.json` but accessible under the versioned API path.
Useful when the well-known path is not accessible.""",
            "operationId": "getAgentDiscoveryApi",
            "responses": {
                "200": _ok_response("Agent card"),
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/a2a/agents": {
        "get": {
            "tags": ["A2A Protocol"],
            "summary": "List available agents",
            "description": """List all agents available through the A2A protocol.

**Response includes:**
- Array of agent cards with capabilities
- Total count of available agents

**Use cases:**
- Discover available agents for task delegation
- Build agent routing tables
- Monitor agent availability""",
            "operationId": "listA2AAgents",
            "responses": {
                "200": {
                    "description": "List of agents",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "agents": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "version": {"type": "string"},
                                                "description": {"type": "string"},
                                                "capabilities": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                            },
                                        },
                                    },
                                    "total": {"type": "integer"},
                                },
                            }
                        }
                    },
                },
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/a2a/agents/{name}": {
        "get": {
            "tags": ["A2A Protocol"],
            "summary": "Get agent by name",
            "description": """Get details for a specific agent by name.

Returns the full agent card including capabilities, version,
and endpoint information.""",
            "operationId": "getA2AAgent",
            "parameters": [
                {
                    "name": "name",
                    "in": "path",
                    "required": True,
                    "description": "Agent name identifier",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Agent details"),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/a2a/tasks": {
        "post": {
            "tags": ["A2A Protocol"],
            "summary": "Submit a task",
            "description": """Submit a task for execution by an A2A agent.

**Required fields:**
- `instruction`: The task instruction to execute

**Optional fields:**
- `task_id`: Custom task ID (auto-generated if not provided)
- `capability`: Preferred capability (debate, audit, critique, research)
- `context`: Additional context items for the task
- `priority`: Task priority level
- `deadline`: ISO 8601 deadline timestamp
- `metadata`: Custom metadata object

**Task execution:**
Tasks are executed asynchronously. Use the task status endpoint
to poll for completion or the stream endpoint for real-time updates.""",
            "operationId": "submitA2ATask",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["instruction"],
                            "properties": {
                                "task_id": {
                                    "type": "string",
                                    "description": "Custom task ID (optional)",
                                },
                                "instruction": {
                                    "type": "string",
                                    "description": "The task instruction to execute",
                                },
                                "capability": {
                                    "type": "string",
                                    "enum": ["debate", "audit", "critique", "research"],
                                    "description": "Preferred agent capability",
                                },
                                "context": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "type": {
                                                "type": "string",
                                                "default": "text",
                                            },
                                            "content": {"type": "string"},
                                            "metadata": {"type": "object"},
                                        },
                                    },
                                    "description": "Additional context items",
                                },
                                "priority": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 10,
                                    "description": "Task priority (1=lowest, 10=highest)",
                                },
                                "deadline": {
                                    "type": "string",
                                    "format": "date-time",
                                    "description": "Task deadline in ISO 8601 format",
                                },
                                "metadata": {
                                    "type": "object",
                                    "description": "Custom metadata",
                                },
                            },
                        },
                        "example": {
                            "instruction": "Analyze the security implications of the proposed API change",
                            "capability": "audit",
                            "context": [
                                {
                                    "type": "text",
                                    "content": "The API will expose user email addresses",
                                }
                            ],
                            "priority": 7,
                        },
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Task result",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "task_id": {"type": "string"},
                                    "status": {
                                        "type": "string",
                                        "enum": [
                                            "pending",
                                            "running",
                                            "completed",
                                            "failed",
                                            "cancelled",
                                        ],
                                    },
                                    "result": {"type": "object"},
                                    "error": {"type": "string"},
                                    "created_at": {"type": "string", "format": "date-time"},
                                    "completed_at": {"type": "string", "format": "date-time"},
                                },
                            }
                        }
                    },
                },
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/a2a/tasks/{task_id}": {
        "get": {
            "tags": ["A2A Protocol"],
            "summary": "Get task status",
            "description": """Get the current status of a submitted task.

**Status values:**
- `pending`: Task queued but not started
- `running`: Task currently executing
- `completed`: Task finished successfully
- `failed`: Task execution failed
- `cancelled`: Task was cancelled

**Polling:** For long-running tasks, poll this endpoint periodically
or use the streaming endpoint for real-time updates.""",
            "operationId": "getA2ATaskStatus",
            "parameters": [
                {
                    "name": "task_id",
                    "in": "path",
                    "required": True,
                    "description": "Task identifier",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": {
                    "description": "Task status",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "task_id": {"type": "string"},
                                    "status": {
                                        "type": "string",
                                        "enum": [
                                            "pending",
                                            "running",
                                            "completed",
                                            "failed",
                                            "cancelled",
                                        ],
                                    },
                                    "result": {"type": "object"},
                                    "error": {"type": "string"},
                                    "progress": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 100,
                                    },
                                    "created_at": {"type": "string", "format": "date-time"},
                                    "started_at": {"type": "string", "format": "date-time"},
                                    "completed_at": {"type": "string", "format": "date-time"},
                                },
                            }
                        }
                    },
                },
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
        "delete": {
            "tags": ["A2A Protocol"],
            "summary": "Cancel task",
            "description": """Cancel a running or pending task.

Only tasks in `pending` or `running` status can be cancelled.
Completed, failed, or already cancelled tasks return a 404 error.

**Graceful cancellation:** Running tasks are given a brief window
to complete their current operation before being forcefully terminated.""",
            "operationId": "cancelA2ATask",
            "parameters": [
                {
                    "name": "task_id",
                    "in": "path",
                    "required": True,
                    "description": "Task identifier",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "204": {"description": "Task cancelled successfully"},
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/a2a/tasks/{task_id}/stream": {
        "post": {
            "tags": ["A2A Protocol"],
            "summary": "Stream task updates",
            "description": """Stream real-time updates for a task.

**Note:** This endpoint requires a WebSocket upgrade.
Returns HTTP 426 with the WebSocket path to connect to.

**WebSocket path:** `/ws/a2a/tasks/{task_id}/stream`

**Stream events:**
- `progress`: Task progress updates
- `log`: Log messages from task execution
- `result`: Final task result
- `error`: Error messages""",
            "operationId": "streamA2ATask",
            "parameters": [
                {
                    "name": "task_id",
                    "in": "path",
                    "required": True,
                    "description": "Task identifier",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "426": {
                    "description": "Upgrade required - Use WebSocket",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "message": {"type": "string"},
                                    "ws_path": {"type": "string"},
                                },
                            },
                            "example": {
                                "message": "Use WebSocket connection for streaming",
                                "ws_path": "/ws/a2a/tasks/task_123/stream",
                            },
                        }
                    },
                },
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/a2a/openapi.json": {
        "get": {
            "tags": ["A2A Protocol"],
            "summary": "Get A2A OpenAPI specification",
            "description": """Get the OpenAPI specification for the A2A protocol endpoints.

This returns a subset of the full Aragora OpenAPI spec containing
only the A2A-related endpoints, useful for generating A2A-specific
client libraries.""",
            "operationId": "getA2AOpenAPISpec",
            "responses": {
                "200": {
                    "description": "OpenAPI specification",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "description": "OpenAPI 3.1 specification object",
                            }
                        }
                    },
                },
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
}
