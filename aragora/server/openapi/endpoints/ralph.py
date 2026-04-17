"""OpenAPI endpoint definitions for Ralph campaign dashboard surfaces."""

from __future__ import annotations

from typing import Any

from aragora.server.openapi.helpers import STANDARD_ERRORS, _ok_response

_obj: dict[str, Any] = {"type": "object", "additionalProperties": True}
_arr_obj: dict[str, Any] = {"type": "array", "items": _obj}
_int: dict[str, Any] = {"type": "integer"}


RALPH_ENDPOINTS: dict[str, dict[str, Any]] = {
    "/api/ralph/campaigns": {
        "get": {
            "tags": ["Ralph"],
            "summary": "List Ralph campaigns",
            "description": "List persisted Ralph campaign supervisor states with summary fields for the campaign dashboard.",
            "operationId": "listRalphCampaigns",
            "responses": {
                "200": _ok_response(
                    "Ralph campaigns",
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "campaigns": _arr_obj,
                            "count": _int,
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "429": STANDARD_ERRORS["429"],
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/ralph/overview": {
        "get": {
            "tags": ["Ralph"],
            "summary": "Get Ralph campaign overview",
            "description": "Return aggregate status, progress, and throughput metrics for Ralph campaign supervisor runs.",
            "operationId": "getRalphOverview",
            "responses": {
                "200": _ok_response(
                    "Ralph campaign overview",
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "data": _obj,
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "429": STANDARD_ERRORS["429"],
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/ralph/blockers": {
        "get": {
            "tags": ["Ralph"],
            "summary": "Get Ralph blocker breakdown",
            "description": "Return aggregate blocker-kind counts and campaign blockage details for Ralph runs.",
            "operationId": "getRalphBlockers",
            "responses": {
                "200": _ok_response(
                    "Ralph blocker breakdown",
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "data": _obj,
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "429": STANDARD_ERRORS["429"],
            },
            "security": [{"bearerAuth": []}],
        }
    },
}
