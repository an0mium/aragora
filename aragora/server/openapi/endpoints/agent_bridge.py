"""OpenAPI endpoint definitions for the agent bridge read API."""

from __future__ import annotations

from typing import Any

from aragora.server.openapi.endpoints.response_schemas import (
    AGENT_BRIDGE_EVENTS_RESPONSE_SCHEMA,
    AGENT_BRIDGE_EVENT_SCHEMA,
    AGENT_BRIDGE_RUN_DETAIL_SCHEMA,
    AGENT_BRIDGE_RUN_LIST_RESPONSE_SCHEMA,
    AGENT_BRIDGE_TRANSCRIPT_RESPONSE_SCHEMA,
)
from aragora.server.openapi.helpers import (
    AUTH_REQUIREMENTS,
    STANDARD_ERRORS,
    STANDARD_RESPONSE_HEADERS,
    _ok_response,
)

_ETAG_HEADER: dict[str, Any] = {
    "description": "Weak entity tag for polling and conditional requests.",
    "schema": {"type": "string"},
}

_RUN_ID_PARAM: dict[str, Any] = {
    "name": "run_id",
    "in": "path",
    "required": True,
    "description": "Bridge run identifier.",
    "schema": {"type": "string"},
}

_LIMIT_PARAM: dict[str, Any] = {
    "name": "limit",
    "in": "query",
    "required": False,
    "description": "Maximum number of records to return per page.",
    "schema": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
}

_CURSOR_PARAM: dict[str, Any] = {
    "name": "cursor",
    "in": "query",
    "required": False,
    "description": "Opaque pagination cursor from a previous response.",
    "schema": {"type": "string"},
}

_NOT_MODIFIED_RESPONSE: dict[str, Any] = {
    "description": "Not modified",
    "headers": {**STANDARD_RESPONSE_HEADERS, "ETag": _ETAG_HEADER},
}


def _secured_get(
    *,
    summary: str,
    description: str,
    operation_id: str,
    responses: dict[str, Any],
    parameters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "tags": ["Agent Bridge"],
        "summary": summary,
        "description": description,
        "operationId": operation_id,
        "security": AUTH_REQUIREMENTS["required"]["security"],
        "parameters": parameters or [],
        "responses": responses,
    }


def _secured_post(
    *,
    summary: str,
    description: str,
    operation_id: str,
    request_schema: dict[str, Any],
    responses: dict[str, Any],
    parameters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "tags": ["Agent Bridge"],
        "summary": summary,
        "description": description,
        "operationId": operation_id,
        "security": AUTH_REQUIREMENTS["required"]["security"],
        "parameters": parameters or [],
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": request_schema}},
        },
        "responses": responses,
    }


_AGENT_BRIDGE_ACTOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["role", "harness"],
    "additionalProperties": False,
    "properties": {
        "role": {"type": "string"},
        "harness": {"type": "string"},
        "model": {"type": "string"},
        "session_id": {"type": "string"},
        "worktree_path": {"type": "string"},
        "worktree_agent_slug": {"type": "string"},
        "branch": {"type": "string"},
        "harness_options": {"type": "object", "additionalProperties": True},
    },
}

_START_RUN_REQUEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["task", "actors"],
    "additionalProperties": False,
    "properties": {
        "task": {"type": "string"},
        "actors": {"type": "array", "items": _AGENT_BRIDGE_ACTOR_SCHEMA, "minItems": 1},
        "run_id": {"type": "string"},
        "next_actor": {"type": "string"},
        "worktree_path": {"type": "string"},
        "worktree_agent_slug": {"type": "string"},
        "repair_budget_per_turn": {"type": "integer", "minimum": 0, "default": 1},
    },
}

_DISPATCH_REQUEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["role", "prompt"],
    "additionalProperties": False,
    "properties": {"role": {"type": "string"}, "prompt": {"type": "string"}},
}

_AUTO_STEP_REQUEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "prompt": {"type": "string"},
        "context_turns": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
    },
}

_AUTO_STEP_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "required": ["auto_step"],
    "properties": {
        **AGENT_BRIDGE_EVENT_SCHEMA["properties"],
        "auto_step": {
            "type": "object",
            "required": ["role", "context_turns"],
            "properties": {
                "role": {"type": "string"},
                "context_turns": {"type": "integer"},
            },
        },
    },
}


AGENT_BRIDGE_ENDPOINTS: dict[str, dict[str, Any]] = {
    "/api/v1/agent-bridge/runs": {
        "get": _secured_get(
            summary="List agent bridge runs",
            description="List persisted agent bridge runs in newest-first order.",
            operation_id="listAgentBridgeRuns",
            parameters=[_LIMIT_PARAM, _CURSOR_PARAM],
            responses={
                "200": _ok_response(
                    "Paginated list of agent bridge runs",
                    AGENT_BRIDGE_RUN_LIST_RESPONSE_SCHEMA,
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        ),
        "post": _secured_post(
            summary="Start agent bridge run",
            description=(
                "Start a persisted bridge run without dispatching a turn. "
                "This operator-local write endpoint is separately feature-gated because "
                "subsequent dispatch can spawn local model harness processes."
            ),
            operation_id="startAgentBridgeRun",
            request_schema=_START_RUN_REQUEST_SCHEMA,
            responses={
                "201": _ok_response("Started agent bridge run", AGENT_BRIDGE_RUN_DETAIL_SCHEMA),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "409": STANDARD_ERRORS["409"],
                "500": STANDARD_ERRORS["500"],
            },
        ),
    },
    "/api/v1/agent-bridge/runs/{run_id}": {
        "get": _secured_get(
            summary="Get agent bridge run",
            description="Return the persisted run summary and role-keyed session registry for a bridge run.",
            operation_id="getAgentBridgeRun",
            parameters=[_RUN_ID_PARAM],
            responses={
                "200": {
                    **_ok_response("Agent bridge run detail", AGENT_BRIDGE_RUN_DETAIL_SCHEMA),
                    "headers": {**STANDARD_RESPONSE_HEADERS, "ETag": _ETAG_HEADER},
                },
                "304": _NOT_MODIFIED_RESPONSE,
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        )
    },
    "/api/v1/agent-bridge/runs/{run_id}/events": {
        "get": _secured_get(
            summary="List agent bridge events",
            description="Return a paginated slice of the persisted agent bridge event stream.",
            operation_id="listAgentBridgeRunEvents",
            parameters=[_RUN_ID_PARAM, _LIMIT_PARAM, _CURSOR_PARAM],
            responses={
                "200": {
                    **_ok_response(
                        "Paginated list of agent bridge events",
                        AGENT_BRIDGE_EVENTS_RESPONSE_SCHEMA,
                    ),
                    "headers": {**STANDARD_RESPONSE_HEADERS, "ETag": _ETAG_HEADER},
                },
                "304": _NOT_MODIFIED_RESPONSE,
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        )
    },
    "/api/v1/agent-bridge/runs/{run_id}/transcript": {
        "get": _secured_get(
            summary="Get agent bridge transcript",
            description="Reconstruct bridge turns from the persisted event log.",
            operation_id="getAgentBridgeRunTranscript",
            parameters=[_RUN_ID_PARAM],
            responses={
                "200": _ok_response(
                    "Reconstructed agent bridge transcript",
                    AGENT_BRIDGE_TRANSCRIPT_RESPONSE_SCHEMA,
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        )
    },
    "/api/v1/agent-bridge/runs/{run_id}/dispatch": {
        "post": _secured_post(
            summary="Dispatch one bridge turn",
            description="Dispatch one prompt to a registered role in an active bridge run.",
            operation_id="dispatchAgentBridgeTurn",
            parameters=[_RUN_ID_PARAM],
            request_schema=_DISPATCH_REQUEST_SCHEMA,
            responses={
                "200": _ok_response("Agent bridge turn event", AGENT_BRIDGE_EVENT_SCHEMA),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "409": STANDARD_ERRORS["409"],
                "500": STANDARD_ERRORS["500"],
            },
        )
    },
    "/api/v1/agent-bridge/runs/{run_id}/auto-step": {
        "post": _secured_post(
            summary="Auto-dispatch the next bridge actor",
            description=(
                "Compose a continuation prompt from the run task and recent transcript, "
                "then dispatch one turn to next_actor. This is one-step automation, not a daemon."
            ),
            operation_id="autoStepAgentBridgeRun",
            parameters=[_RUN_ID_PARAM],
            request_schema=_AUTO_STEP_REQUEST_SCHEMA,
            responses={
                "200": _ok_response("Auto-step dispatch result", _AUTO_STEP_RESPONSE_SCHEMA),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "409": STANDARD_ERRORS["409"],
                "500": STANDARD_ERRORS["500"],
            },
        )
    },
}

__all__ = ["AGENT_BRIDGE_ENDPOINTS"]
