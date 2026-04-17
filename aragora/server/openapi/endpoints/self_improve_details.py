"""OpenAPI endpoint definitions for self-improvement detail surfaces."""

from __future__ import annotations

from typing import Any

from aragora.server.openapi.helpers import STANDARD_ERRORS, _ok_response

_obj: dict[str, Any] = {"type": "object", "additionalProperties": True}
_arr_obj: dict[str, Any] = {"type": "array", "items": _obj}


def _json_request(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": True,
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return {
        "required": True,
        "content": {"application/json": {"schema": schema}},
    }


def _path_param(name: str, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "in": "path",
        "required": True,
        "schema": {"type": "string"},
        "description": description,
    }


_DETAIL_RESPONSE = {
    "type": "object",
    "additionalProperties": True,
    "properties": {"data": _obj},
}

_QUEUE_ITEM_RESPONSE = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "data": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "id": {"type": "string"},
                "goal": {"type": "string"},
                "priority": {"type": "integer"},
                "source": {"type": "string"},
                "status": {"type": "string"},
                "createdAt": {"type": "string"},
            },
        }
    },
}

_QUEUE_UNAVAILABLE = {
    "description": "Improvement queue unavailable",
    "content": {
        "application/json": {
            "schema": {
                "type": "object",
                "additionalProperties": True,
                "properties": {"error": {"type": "string"}},
            }
        }
    },
}


SELF_IMPROVE_DETAILS_ENDPOINTS: dict[str, dict[str, Any]] = {
    "/api/self-improve/meta-planner/goals": {
        "get": {
            "tags": ["Nomic"],
            "summary": "List self-improvement meta-planner goals",
            "description": "Return prioritized MetaPlanner goals, codebase signals, and queue summary data for the self-improvement dashboard.",
            "operationId": "listSelfImproveMetaPlannerGoals",
            "responses": {
                "200": _ok_response("Meta-planner goals", _DETAIL_RESPONSE),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/self-improve/execution/timeline": {
        "get": {
            "tags": ["Nomic"],
            "summary": "List self-improvement execution timeline",
            "description": "Return active self-improvement branches, merge decisions, and execution timeline data.",
            "operationId": "listSelfImproveExecutionTimeline",
            "responses": {
                "200": _ok_response("Execution timeline", _DETAIL_RESPONSE),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/self-improve/learning/insights": {
        "get": {
            "tags": ["Nomic"],
            "summary": "List self-improvement learning insights",
            "description": "Return cross-cycle learning insights and recurring self-improvement patterns.",
            "operationId": "listSelfImproveLearningInsights",
            "responses": {
                "200": _ok_response("Learning insights", _DETAIL_RESPONSE),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/self-improve/metrics/comparison": {
        "get": {
            "tags": ["Nomic"],
            "summary": "Get self-improvement metrics comparison",
            "description": "Return before/after codebase metrics and regression comparisons across self-improvement cycles.",
            "operationId": "getSelfImproveMetricsComparison",
            "responses": {
                "200": _ok_response("Metrics comparison", _DETAIL_RESPONSE),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/self-improve/trends/cycles": {
        "get": {
            "tags": ["Nomic"],
            "summary": "List self-improvement cycle trends",
            "description": "Return cycle trends, aggregate success metrics, and run cost summaries.",
            "operationId": "listSelfImproveCycleTrends",
            "responses": {
                "200": _ok_response(
                    "Cycle trends",
                    {
                        "type": "object",
                        "additionalProperties": True,
                        "properties": {
                            "data": {
                                "type": "object",
                                "additionalProperties": True,
                                "properties": {
                                    "cycles": _arr_obj,
                                    "summary": _obj,
                                    "run_costs": _arr_obj,
                                },
                            }
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/self-improve/improvement-queue": {
        "post": {
            "tags": ["Nomic"],
            "summary": "Create self-improvement queue item",
            "description": "Add a user-submitted goal to the self-improvement queue.",
            "operationId": "createSelfImproveQueueItem",
            "requestBody": _json_request(
                {
                    "goal": {
                        "type": "string",
                        "description": "Improvement goal to add to the queue.",
                    },
                    "priority": {
                        "type": "integer",
                        "default": 50,
                        "minimum": 0,
                        "maximum": 100,
                    },
                    "source": {"type": "string", "default": "user"},
                },
                ["goal"],
            ),
            "responses": {
                "201": _ok_response("Queued improvement goal", _QUEUE_ITEM_RESPONSE),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "503": _QUEUE_UNAVAILABLE,
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/self-improve/improvement-queue/{id}/priority": {
        "put": {
            "tags": ["Nomic"],
            "summary": "Update self-improvement queue priority",
            "description": "Update the priority for a self-improvement queue item.",
            "operationId": "updateSelfImproveQueueItemPriority",
            "parameters": [_path_param("id", "Self-improvement queue item id.")],
            "requestBody": _json_request(
                {
                    "priority": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                    }
                },
                ["priority"],
            ),
            "responses": {
                "200": _ok_response("Updated queue item priority", _QUEUE_ITEM_RESPONSE),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "503": _QUEUE_UNAVAILABLE,
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/self-improve/improvement-queue/{id}": {
        "delete": {
            "tags": ["Nomic"],
            "summary": "Delete self-improvement queue item",
            "description": "Remove an item from the self-improvement queue.",
            "operationId": "deleteSelfImproveQueueItem",
            "parameters": [_path_param("id", "Self-improvement queue item id.")],
            "responses": {
                "200": _ok_response("Deleted queue item", _QUEUE_ITEM_RESPONSE),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "503": _QUEUE_UNAVAILABLE,
            },
            "security": [{"bearerAuth": []}],
        }
    },
}
