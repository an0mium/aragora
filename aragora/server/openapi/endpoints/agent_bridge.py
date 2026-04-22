"""OpenAPI endpoint definitions for the agent bridge read API."""

from __future__ import annotations

from typing import Any

from aragora.server.openapi.endpoints.response_schemas import (
    AGENT_BRIDGE_EVENTS_RESPONSE_SCHEMA,
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
        )
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
}

__all__ = ["AGENT_BRIDGE_ENDPOINTS"]
