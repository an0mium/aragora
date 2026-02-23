"""
OpenAPI endpoint definitions for Computer Use.

Endpoints for managing computer use tasks, actions, and policies
for AI-driven computer interaction orchestration.
"""

from typing import Any

from aragora.server.openapi.helpers import STANDARD_ERRORS


# Helper to build inline response
def _response(description: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a response with optional inline schema."""
    resp: dict[str, Any] = {"description": description}
    if schema:
        resp["content"] = {"application/json": {"schema": schema}}
    return resp


COMPUTER_USE_ENDPOINTS = {
    # =========================================================================
    # Actions
    # =========================================================================
    "/api/v1/computer-use/actions": {
        "get": {
            "tags": ["Computer Use"],
            "summary": "List computer use actions",
            "description": "List recent computer use actions with statistics. Returns aggregated action counts by type (click, type, screenshot, scroll, key) across all completed tasks.",
            "operationId": "listComputerUseActions",
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of actions to return (default: 20)",
                    "schema": {"type": "integer", "default": 20},
                },
                {
                    "name": "offset",
                    "in": "query",
                    "description": "Pagination offset",
                    "schema": {"type": "integer", "default": 0},
                },
                {
                    "name": "action_type",
                    "in": "query",
                    "description": "Filter by action type",
                    "schema": {
                        "type": ["string", "null"],
                        "enum": ["click", "type", "screenshot", "scroll", "key"],
                    },
                },
            ],
            "responses": {
                "200": _response(
                    "List of computer use actions",
                    {
                        "type": ["object", "null"],
                        "properties": {
                            "actions": {
                                "type": "array",
                                "items": {
                                    "type": ["object", "null"],
                                    "properties": {
                                        "action_id": {"type": "string"},
                                        "action_type": {
                                            "type": "string",
                                            "enum": [
                                                "click",
                                                "type",
                                                "screenshot",
                                                "scroll",
                                                "key",
                                            ],
                                        },
                                        "task_id": {"type": "string"},
                                        "success": {"type": "boolean"},
                                        "created_at": {"type": "string", "format": "date-time"},
                                    },
                                },
                            },
                            "total": {"type": "integer"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
        "post": {
            "tags": ["Computer Use"],
            "summary": "Execute a computer use action",
            "description": "Execute a single computer use action (click, type, screenshot, scroll, key). The action is validated against the active policy before execution.",
            "operationId": "createComputerUseAction",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["action_type"],
                            "properties": {
                                "action_type": {
                                    "type": "string",
                                    "enum": ["click", "type", "screenshot", "scroll", "key"],
                                    "description": "The type of action to perform",
                                },
                                "parameters": {
                                    "type": "object",
                                    "description": "Action-specific parameters (e.g., coordinates for click, text for type)",
                                    "properties": {
                                        "x": {
                                            "type": "integer",
                                            "description": "X coordinate (for click)",
                                        },
                                        "y": {
                                            "type": "integer",
                                            "description": "Y coordinate (for click)",
                                        },
                                        "text": {
                                            "type": "string",
                                            "description": "Text to type (for type action)",
                                        },
                                        "key": {
                                            "type": "string",
                                            "description": "Key to press (for key action)",
                                        },
                                        "direction": {
                                            "type": "string",
                                            "enum": ["up", "down"],
                                            "description": "Scroll direction (for scroll action)",
                                        },
                                        "amount": {
                                            "type": "integer",
                                            "description": "Scroll amount (for scroll action)",
                                        },
                                    },
                                },
                                "task_id": {
                                    "type": "string",
                                    "description": "Associated task ID (optional)",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "201": _response(
                    "Action executed",
                    {
                        "type": "object",
                        "properties": {
                            "action_id": {"type": "string"},
                            "action_type": {"type": "string"},
                            "success": {"type": "boolean"},
                            "message": {"type": "string"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/computer-use/actions/{action_id}": {
        "get": {
            "tags": ["Computer Use"],
            "summary": "Get action details",
            "description": "Get detailed information about a specific computer use action, including its parameters, result, and associated task.",
            "operationId": "getComputerUseAction",
            "parameters": [
                {
                    "name": "action_id",
                    "in": "path",
                    "required": True,
                    "description": "Action ID",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _response(
                    "Action details",
                    {
                        "type": "object",
                        "properties": {
                            "action_id": {"type": "string"},
                            "action_type": {
                                "type": "string",
                                "enum": ["click", "type", "screenshot", "scroll", "key"],
                            },
                            "task_id": {"type": "string"},
                            "parameters": {"type": "object"},
                            "success": {"type": "boolean"},
                            "result": {"type": ["object", "null"]},
                            "created_at": {"type": "string", "format": "date-time"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
        "delete": {
            "tags": ["Computer Use"],
            "summary": "Delete an action record",
            "description": "Delete a computer use action record. Only completed or failed actions can be deleted.",
            "operationId": "deleteComputerUseAction",
            "parameters": [
                {
                    "name": "action_id",
                    "in": "path",
                    "required": True,
                    "description": "Action ID",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _response(
                    "Action deleted",
                    {
                        "type": "object",
                        "properties": {
                            "deleted": {"type": "boolean"},
                            "action_id": {"type": "string"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    # =========================================================================
    # Policies
    # =========================================================================
    "/api/v1/computer-use/policies": {
        "get": {
            "tags": ["Computer Use"],
            "summary": "List computer use policies",
            "description": "List all active computer use policies. Policies define which actions are allowed, blocked domains, and execution constraints.",
            "operationId": "listComputerUsePolicies",
            "responses": {
                "200": _response(
                    "List of policies",
                    {
                        "type": "object",
                        "properties": {
                            "policies": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "name": {"type": "string"},
                                        "description": {"type": "string"},
                                        "allowed_actions": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "blocked_domains": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                },
                            },
                            "total": {"type": "integer"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
        "post": {
            "tags": ["Computer Use"],
            "summary": "Create a computer use policy",
            "description": "Create a new computer use policy that defines allowed actions, blocked domains, and execution constraints for computer use tasks.",
            "operationId": "createComputerUsePolicy",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["name"],
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Policy name",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Policy description",
                                },
                                "allowed_actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "enum": ["click", "type", "screenshot", "scroll", "key"],
                                    },
                                    "description": "List of allowed action types",
                                },
                                "blocked_domains": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of domains to block",
                                },
                                "max_steps": {
                                    "type": "integer",
                                    "description": "Maximum number of steps per task",
                                    "default": 20,
                                },
                                "timeout_seconds": {
                                    "type": "integer",
                                    "description": "Timeout in seconds for task execution",
                                    "default": 300,
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "201": _response(
                    "Policy created",
                    {
                        "type": "object",
                        "properties": {
                            "policy_id": {"type": "string"},
                            "message": {"type": "string"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/computer-use/policies/{policy_id}": {
        "get": {
            "tags": ["Computer Use"],
            "summary": "Get policy details",
            "description": "Get detailed information about a specific computer use policy, including its allowed actions, blocked domains, and constraints.",
            "operationId": "getComputerUsePolicy",
            "parameters": [
                {
                    "name": "policy_id",
                    "in": "path",
                    "required": True,
                    "description": "Policy ID",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _response(
                    "Policy details",
                    {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "allowed_actions": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "blocked_domains": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "max_steps": {"type": "integer"},
                            "timeout_seconds": {"type": "integer"},
                            "created_at": {"type": "string", "format": "date-time"},
                            "updated_at": {"type": "string", "format": "date-time"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
        "put": {
            "tags": ["Computer Use"],
            "summary": "Update a computer use policy",
            "description": "Replace the configuration of an existing computer use policy.",
            "operationId": "updateComputerUsePolicy",
            "parameters": [
                {
                    "name": "policy_id",
                    "in": "path",
                    "required": True,
                    "description": "Policy ID",
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
                                "name": {
                                    "type": "string",
                                    "description": "Policy name",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Policy description",
                                },
                                "allowed_actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "enum": ["click", "type", "screenshot", "scroll", "key"],
                                    },
                                    "description": "List of allowed action types",
                                },
                                "blocked_domains": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of domains to block",
                                },
                                "max_steps": {
                                    "type": "integer",
                                    "description": "Maximum number of steps per task",
                                },
                                "timeout_seconds": {
                                    "type": "integer",
                                    "description": "Timeout in seconds for task execution",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _response(
                    "Policy updated",
                    {
                        "type": "object",
                        "properties": {
                            "policy_id": {"type": "string"},
                            "message": {"type": "string"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
        "delete": {
            "tags": ["Computer Use"],
            "summary": "Delete a computer use policy",
            "description": "Delete a computer use policy. The default policy cannot be deleted.",
            "operationId": "deleteComputerUsePolicy",
            "parameters": [
                {
                    "name": "policy_id",
                    "in": "path",
                    "required": True,
                    "description": "Policy ID",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _response(
                    "Policy deleted",
                    {
                        "type": "object",
                        "properties": {
                            "deleted": {"type": "boolean"},
                            "policy_id": {"type": "string"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    # =========================================================================
    # Tasks
    # =========================================================================
    "/api/v1/computer-use/tasks": {
        "get": {
            "tags": ["Computer Use"],
            "summary": "List computer use tasks",
            "description": "List recent computer use tasks. Tasks represent high-level goals that are decomposed into individual actions by the orchestrator.",
            "operationId": "listComputerUseTasks",
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of tasks to return (default: 20)",
                    "schema": {"type": "integer", "default": 20},
                },
                {
                    "name": "status",
                    "in": "query",
                    "description": "Filter by task status",
                    "schema": {
                        "type": "string",
                        "enum": ["pending", "running", "completed", "failed", "cancelled"],
                    },
                },
            ],
            "responses": {
                "200": _response(
                    "List of tasks",
                    {
                        "type": "object",
                        "properties": {
                            "tasks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "task_id": {"type": "string"},
                                        "goal": {"type": "string"},
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
                                        "max_steps": {"type": "integer"},
                                        "dry_run": {"type": "boolean"},
                                        "created_at": {"type": "string", "format": "date-time"},
                                        "result": {
                                            "type": ["object", "null"],
                                            "properties": {
                                                "success": {"type": "boolean"},
                                                "message": {"type": "string"},
                                                "steps_taken": {"type": "integer"},
                                            },
                                        },
                                    },
                                },
                            },
                            "total": {"type": "integer"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
        "post": {
            "tags": ["Computer Use"],
            "summary": "Create a computer use task",
            "description": "Create and run a computer use task. The orchestrator will decompose the goal into individual actions and execute them step by step.",
            "operationId": "createComputerUseTask",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["goal"],
                            "properties": {
                                "goal": {
                                    "type": "string",
                                    "description": "The goal to accomplish via computer use",
                                },
                                "max_steps": {
                                    "type": "integer",
                                    "description": "Maximum number of steps to execute (default: 10)",
                                    "default": 10,
                                },
                                "dry_run": {
                                    "type": "boolean",
                                    "description": "If true, simulate execution without performing actions",
                                    "default": False,
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "201": _response(
                    "Task created",
                    {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "running", "completed", "failed"],
                            },
                            "message": {"type": "string"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "429": STANDARD_ERRORS["429"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/computer-use/tasks/{task_id}": {
        "get": {
            "tags": ["Computer Use"],
            "summary": "Get task details",
            "description": "Get detailed information about a specific computer use task, including its goal, status, steps executed, and result.",
            "operationId": "getComputerUseTask",
            "parameters": [
                {
                    "name": "task_id",
                    "in": "path",
                    "required": True,
                    "description": "Task ID",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _response(
                    "Task details",
                    {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "object",
                                "properties": {
                                    "task_id": {"type": "string"},
                                    "goal": {"type": "string"},
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
                                    "max_steps": {"type": "integer"},
                                    "dry_run": {"type": "boolean"},
                                    "created_at": {"type": "string", "format": "date-time"},
                                    "cancelled_at": {
                                        "type": ["string", "null"],
                                        "format": "date-time",
                                    },
                                    "steps": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "action": {"type": "string"},
                                                "success": {"type": "boolean"},
                                            },
                                        },
                                    },
                                    "result": {
                                        "type": ["object", "null"],
                                        "properties": {
                                            "success": {"type": "boolean"},
                                            "message": {"type": "string"},
                                            "steps_taken": {"type": "integer"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
        "delete": {
            "tags": ["Computer Use"],
            "summary": "Delete a task record",
            "description": "Delete a computer use task record. Only completed, failed, or cancelled tasks can be deleted. Running tasks must be cancelled first.",
            "operationId": "deleteComputerUseTask",
            "parameters": [
                {
                    "name": "task_id",
                    "in": "path",
                    "required": True,
                    "description": "Task ID",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _response(
                    "Task deleted",
                    {
                        "type": "object",
                        "properties": {
                            "deleted": {"type": "boolean"},
                            "task_id": {"type": "string"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/computer-use/approvals": {
        "get": {
            "tags": ["Computer Use"],
            "summary": "List computer-use approvals",
            "description": "List approval requests for sensitive computer-use actions.",
            "operationId": "listComputerUseApprovals",
            "parameters": [
                {
                    "name": "status",
                    "in": "query",
                    "description": "Filter by approval status",
                    "schema": {
                        "type": "string",
                        "enum": ["pending", "approved", "denied", "expired", "cancelled"],
                    },
                },
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of approvals to return (default: 50)",
                    "schema": {"type": "integer", "default": 50},
                },
            ],
            "responses": {
                "200": _response(
                    "List of approval requests",
                    {
                        "type": "object",
                        "properties": {
                            "approvals": {
                                "type": "array",
                                "items": {"type": "object"},
                            },
                            "count": {"type": "integer"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/computer-use/approvals/{request_id}": {
        "get": {
            "tags": ["Computer Use"],
            "summary": "Get approval request",
            "description": "Get details for a specific computer-use approval request.",
            "operationId": "getComputerUseApproval",
            "parameters": [
                {
                    "name": "request_id",
                    "in": "path",
                    "required": True,
                    "description": "Approval request ID",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _response(
                    "Approval request",
                    {
                        "type": "object",
                        "properties": {
                            "approval": {"type": "object"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/computer-use/approvals/{request_id}/approve": {
        "post": {
            "tags": ["Computer Use"],
            "summary": "Approve a computer-use request",
            "description": "Approve a pending computer-use approval request.",
            "operationId": "approveComputerUseRequest",
            "parameters": [
                {
                    "name": "request_id",
                    "in": "path",
                    "required": True,
                    "description": "Approval request ID",
                    "schema": {"type": "string"},
                }
            ],
            "requestBody": {
                "required": False,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "reason": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _response(
                    "Approval granted",
                    {
                        "type": "object",
                        "properties": {
                            "approved": {"type": "boolean"},
                            "request_id": {"type": "string"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/computer-use/approvals/{request_id}/deny": {
        "post": {
            "tags": ["Computer Use"],
            "summary": "Deny a computer-use request",
            "description": "Deny a pending computer-use approval request.",
            "operationId": "denyComputerUseRequest",
            "parameters": [
                {
                    "name": "request_id",
                    "in": "path",
                    "required": True,
                    "description": "Approval request ID",
                    "schema": {"type": "string"},
                }
            ],
            "requestBody": {
                "required": False,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "reason": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _response(
                    "Approval denied",
                    {
                        "type": "object",
                        "properties": {
                            "denied": {"type": "boolean"},
                            "request_id": {"type": "string"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
}
