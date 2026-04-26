"""OpenAPI endpoint definitions for Ralph campaign dashboard surfaces."""

from __future__ import annotations

from typing import Any

from aragora.server.openapi.helpers import STANDARD_ERRORS, _ok_response

_obj: dict[str, Any] = {"type": "object", "additionalProperties": True}
_arr_obj: dict[str, Any] = {"type": "array", "items": _obj}
_int: dict[str, Any] = {"type": "integer"}


def _campaign_id_parameter() -> dict[str, Any]:
    return {
        "name": "campaign_id",
        "in": "path",
        "required": True,
        "description": "Ralph campaign identifier",
        "schema": {"type": "string"},
    }


def _campaign_get_endpoint(
    *,
    summary: str,
    description: str,
    operation_id: str,
    response_name: str,
    response_schema: dict[str, Any],
) -> dict[str, Any]:
    return {
        "get": {
            "tags": ["Ralph"],
            "summary": summary,
            "description": description,
            "operationId": operation_id,
            "parameters": [_campaign_id_parameter()],
            "responses": {
                "200": _ok_response(
                    summary,
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            response_name: response_schema,
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "429": STANDARD_ERRORS["429"],
            },
            "security": [{"bearerAuth": []}],
        }
    }


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
    "/api/ralph/campaigns/{campaign_id}": _campaign_get_endpoint(
        summary="Get Ralph campaign detail",
        description="Return detailed state, progress, and health data for a Ralph campaign.",
        operation_id="getRalphCampaign",
        response_name="data",
        response_schema=_obj,
    ),
    "/api/ralph/campaigns/{campaign_id}/timeline": _campaign_get_endpoint(
        summary="Get Ralph campaign timeline",
        description="Return the step timeline for a Ralph campaign supervisor run.",
        operation_id="getRalphCampaignTimeline",
        response_name="timeline",
        response_schema=_arr_obj,
    ),
    "/api/ralph/campaigns/{campaign_id}/blockers": _campaign_get_endpoint(
        summary="Get Ralph campaign blockers",
        description="Return blocker breakdown data for a specific Ralph campaign.",
        operation_id="getRalphCampaignBlockers",
        response_name="data",
        response_schema=_obj,
    ),
    "/api/ralph/campaigns/{campaign_id}/repairs": _campaign_get_endpoint(
        summary="Get Ralph campaign repairs",
        description="Return repair statistics for a specific Ralph campaign.",
        operation_id="getRalphCampaignRepairs",
        response_name="data",
        response_schema=_obj,
    ),
    "/api/ralph/campaigns/{campaign_id}/budget": _campaign_get_endpoint(
        summary="Get Ralph campaign budget",
        description="Return budget burn and limit data for a specific Ralph campaign.",
        operation_id="getRalphCampaignBudget",
        response_name="data",
        response_schema=_obj,
    ),
    "/api/ralph/campaigns/{campaign_id}/pr-gate": _campaign_get_endpoint(
        summary="Get Ralph campaign PR gate",
        description="Return PR merge gate readiness for a specific Ralph campaign.",
        operation_id="getRalphCampaignPrGate",
        response_name="data",
        response_schema=_obj,
    ),
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
