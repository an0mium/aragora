"""OpenAPI endpoint definitions for feedback-hub visibility routes."""

from __future__ import annotations

from typing import Any

from aragora.server.openapi.helpers import STANDARD_ERRORS, _ok_response

_str: dict[str, Any] = {"type": "string"}
_int: dict[str, Any] = {"type": "integer"}
_num: dict[str, Any] = {"type": "number"}
_bool: dict[str, Any] = {"type": "boolean"}
_count_map: dict[str, Any] = {
    "type": "object",
    "additionalProperties": {"type": "integer"},
}

_HUB_STATS: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "total_routed": _int,
        "total_failures": _int,
        "by_source": _count_map,
        "by_target": _count_map,
        "history_size": _int,
        "known_sources": {"type": "array", "items": _str},
    },
}

_HUB_HISTORY_ENTRY: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "source": _str,
        "targets_hit": {"type": "array", "items": _str},
        "targets_failed": {"type": "array", "items": _str},
        "errors": {"type": "array", "items": _str},
        "routed_at": _num,
        "success": _bool,
    },
}

_STATS_ENVELOPE: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["data"],
    "properties": {"data": _HUB_STATS},
}

_HISTORY_ENVELOPE: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["data"],
    "properties": {"data": {"type": "array", "items": _HUB_HISTORY_ENTRY}},
}

_HISTORY_LIMIT_PARAM: dict[str, Any] = {
    "name": "limit",
    "in": "query",
    "description": "Maximum number of feedback-routing history entries to return.",
    "schema": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
}


def _stats_operation(operation_id: str) -> dict[str, Any]:
    return {
        "tags": ["Feedback Hub"],
        "summary": "Get feedback-hub routing stats",
        "description": "Return counters for feedback signals routed through the unified self-improvement feedback hub.",
        "operationId": operation_id,
        "responses": {
            "200": _ok_response("Feedback-hub routing statistics", _STATS_ENVELOPE),
            "401": STANDARD_ERRORS["401"],
            "403": STANDARD_ERRORS["403"],
            "429": STANDARD_ERRORS["429"],
            "503": STANDARD_ERRORS["500"],
        },
        "security": [{"bearerAuth": ["admin"]}],
    }


def _history_operation(operation_id: str) -> dict[str, Any]:
    return {
        "tags": ["Feedback Hub"],
        "summary": "List feedback-hub routing history",
        "description": "Return recent feedback-routing decisions made by the unified self-improvement feedback hub.",
        "operationId": operation_id,
        "parameters": [_HISTORY_LIMIT_PARAM],
        "responses": {
            "200": _ok_response("Feedback-hub routing history", _HISTORY_ENVELOPE),
            "401": STANDARD_ERRORS["401"],
            "403": STANDARD_ERRORS["403"],
            "429": STANDARD_ERRORS["429"],
            "503": STANDARD_ERRORS["500"],
        },
        "security": [{"bearerAuth": ["admin"]}],
    }


FEEDBACK_HUB_ENDPOINTS: dict[str, dict[str, Any]] = {
    "/api/feedback-hub/stats": {
        "get": _stats_operation("getFeedbackHubStats"),
    },
    "/api/feedback-hub/history": {
        "get": _history_operation("listFeedbackHubHistory"),
    },
    "/api/v1/feedback-hub/stats": {
        "get": _stats_operation("getFeedbackHubStatsV1"),
    },
    "/api/v1/feedback-hub/history": {
        "get": _history_operation("listFeedbackHubHistoryV1"),
    },
}
