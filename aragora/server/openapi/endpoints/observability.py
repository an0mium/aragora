"""OpenAPI endpoint definitions for observability surfaces."""

from __future__ import annotations

from typing import Any

from aragora.server.openapi.helpers import STANDARD_ERRORS, _ok_response

_obj: dict[str, Any] = {"type": "object", "additionalProperties": True}
_str: dict[str, Any] = {"type": "string"}
_int: dict[str, Any] = {"type": "integer"}
_num: dict[str, Any] = {"type": "number"}
_arr_obj: dict[str, Any] = {"type": "array", "items": _obj}


def _json_request(
    properties: dict[str, Any], *, required: list[str] | None = None
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": True,
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return {"required": True, "content": {"application/json": {"schema": schema}}}


_CRASH_REPORT = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "message": _str,
        "stack": _str,
        "componentStack": _str,
        "url": _str,
        "timestamp": _str,
        "userAgent": _str,
        "sessionId": _str,
        "componentName": _str,
        "fingerprint": _str,
    },
}


OBSERVABILITY_ENDPOINTS: dict[str, dict[str, Any]] = {
    "/api/observability/dashboard": {
        "get": {
            "tags": ["Observability"],
            "summary": "Get observability dashboard",
            "description": "Return aggregated debate, agent, circuit-breaker, self-improvement, and health metrics for the operator dashboard.",
            "operationId": "getObservabilityDashboard",
            "responses": {
                "200": _ok_response(
                    "Aggregated observability dashboard",
                    {
                        "type": "object",
                        "additionalProperties": True,
                        "properties": {
                            "timestamp": _num,
                            "debate_metrics": _obj,
                            "agent_rankings": _obj,
                            "circuit_breakers": _obj,
                            "self_improve": _obj,
                            "alerts": _arr_obj,
                            "system_health": _obj,
                            "error_rates": _obj,
                            "collection_time_ms": _num,
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
    "/api/observability/metrics": {
        "get": {
            "tags": ["Observability"],
            "summary": "Get observability metrics",
            "description": "Return the same aggregated metrics payload used by the observability dashboard.",
            "operationId": "getObservabilityMetrics",
            "responses": {
                "200": _ok_response("Aggregated observability metrics", _obj),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "429": STANDARD_ERRORS["429"],
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/observability/crashes": {
        "get": {
            "tags": ["Observability"],
            "summary": "List crash telemetry",
            "description": "List recent frontend crash telemetry reports for administrative review.",
            "operationId": "listObservabilityCrashes",
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                },
                {
                    "name": "offset",
                    "in": "query",
                    "schema": {"type": "integer", "minimum": 0, "default": 0},
                },
            ],
            "responses": {
                "200": _ok_response(
                    "Crash telemetry page",
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "crashes": _arr_obj,
                            "total": _int,
                            "limit": _int,
                            "offset": _int,
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "429": STANDARD_ERRORS["429"],
            },
            "security": [{"bearerAuth": ["admin"]}],
        },
        "post": {
            "tags": ["Observability"],
            "summary": "Ingest crash telemetry",
            "description": "Accept a bounded batch of frontend crash reports and store new fingerprints for later review.",
            "operationId": "createObservabilityCrashes",
            "requestBody": _json_request(
                {
                    "reports": {
                        "type": "array",
                        "items": _CRASH_REPORT,
                        "maxItems": 50,
                    }
                },
                required=["reports"],
            ),
            "responses": {
                "202": _ok_response(
                    "Crash telemetry accepted",
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "accepted": _int,
                            "duplicates": _int,
                            "total_stored": _int,
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "429": STANDARD_ERRORS["429"],
            },
            "security": [],
        },
    },
    "/api/observability/crashes/stats": {
        "get": {
            "tags": ["Observability"],
            "summary": "Get crash telemetry stats",
            "description": "Return aggregate crash telemetry counters, top fingerprints, and component hot spots.",
            "operationId": "getObservabilityCrashStats",
            "responses": {
                "200": _ok_response(
                    "Crash telemetry stats",
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "total_ingested": _int,
                            "total_stored": _int,
                            "total_duplicates": _int,
                            "total_rate_limited": _int,
                            "unique_fingerprints": _int,
                            "last_hour": _int,
                            "last_24h": _int,
                            "top_fingerprints": _arr_obj,
                            "top_components": _arr_obj,
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "429": STANDARD_ERRORS["429"],
            },
            "security": [{"bearerAuth": ["admin"]}],
        }
    },
}
