"""Playground and spectate public-surface endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

_obj = {"type": "object"}
_str = {"type": "string"}
_num = {"type": "number"}
_int = {"type": "integer"}
_bool = {"type": "boolean"}


PLAYGROUND_ENDPOINTS: dict = {
    "/api/v1/playground/assess": {
        "post": {
            "tags": ["Debates"],
            "summary": "Assess landing question clarity",
            "description": "Check whether a landing-page question is ready for debate or needs clarification.",
            "operationId": "createPlaygroundAssessment",
            "responses": {
                "200": _ok_response(
                    "Assessment result",
                    {
                        "type": "object",
                        "properties": {
                            "type": _str,
                            "option": _obj,
                            "preflight": _obj,
                            "error": _str,
                            "code": _str,
                            "retry_after": _int,
                        },
                    },
                    include_rate_limit_headers=True,
                )
            },
        }
    },
    "/api/v1/playground/landing/events": {
        "post": {
            "tags": ["Debates"],
            "summary": "Record landing telemetry event",
            "description": "Capture bounded telemetry from the public landing page.",
            "operationId": "createPlaygroundLandingEvent",
            "responses": {
                "202": _ok_response(
                    "Telemetry accepted",
                    {
                        "type": "object",
                        "properties": {
                            "ok": _bool,
                        },
                    },
                )
            },
        }
    },
    "/api/v1/playground/landing/events/summary": {
        "get": {
            "tags": ["Debates"],
            "summary": "Summarize recent landing telemetry",
            "description": "Aggregate recent landing-page telemetry without exposing raw events.",
            "operationId": "getPlaygroundLandingEventSummary",
            "parameters": [
                {
                    "name": "window",
                    "in": "query",
                    "description": "Lookback window in seconds.",
                    "schema": {
                        "type": "number",
                        "default": 86400,
                        "minimum": 60,
                        "maximum": 604800,
                    },
                },
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of top options to include.",
                    "schema": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
                },
            ],
            "responses": {"200": _ok_response("Landing telemetry summary", _obj)},
        }
    },
    "/api/v1/playground/landing/feedback": {
        "get": {
            "tags": ["Debates"],
            "summary": "List landing feedback reports",
            "description": "List recent public landing-page wrong-answer reports for admins.",
            "operationId": "listPlaygroundLandingFeedback",
            "parameters": [
                {
                    "name": "window",
                    "in": "query",
                    "description": "Lookback window in seconds.",
                    "schema": {
                        "type": "number",
                        "default": 604800,
                        "minimum": 300,
                        "maximum": 2592000,
                    },
                },
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of reports to return.",
                    "schema": {"type": "integer", "default": 50, "minimum": 1, "maximum": 200},
                },
            ],
            "responses": {"200": _ok_response("Landing feedback summary", _obj)},
            "security": [{"bearerAuth": []}],
        },
        "post": {
            "tags": ["Debates"],
            "summary": "Submit landing feedback report",
            "description": "Capture a bounded wrong-answer report from the public landing page.",
            "operationId": "createPlaygroundLandingFeedback",
            "responses": {
                "202": _ok_response(
                    "Feedback accepted",
                    {
                        "type": "object",
                        "properties": {
                            "ok": _bool,
                            "report_id": _str,
                        },
                    },
                )
            },
        },
    },
    "/api/v1/playground/landing/feedback/review": {
        "post": {
            "tags": ["Debates"],
            "summary": "Review landing feedback report",
            "description": "Update admin review state for a public landing-page feedback report.",
            "operationId": "createPlaygroundLandingFeedbackReview",
            "responses": {
                "200": _ok_response(
                    "Review state updated",
                    {
                        "type": "object",
                        "properties": {
                            "ok": _bool,
                            "id": _str,
                            "review_status": _str,
                            "reviewed_at": {"type": ["string", "null"], "format": "date-time"},
                            "reviewed_by": {"type": ["string", "null"]},
                        },
                    },
                )
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/v1/spectate/recent": {
        "get": {
            "tags": ["Debates"],
            "summary": "List recent spectate events",
            "description": "Return buffered spectate events, optionally filtered to a debate or pipeline.",
            "operationId": "getSpectateRecent",
            "parameters": [
                {
                    "name": "count",
                    "in": "query",
                    "description": "Maximum number of buffered events to return.",
                    "schema": {"type": "integer", "default": 50, "minimum": 1, "maximum": 500},
                },
                {
                    "name": "debate_id",
                    "in": "query",
                    "description": "Limit results to one debate.",
                    "schema": _str,
                },
                {
                    "name": "pipeline_id",
                    "in": "query",
                    "description": "Limit results to one pipeline.",
                    "schema": _str,
                },
            ],
            "responses": {"200": _ok_response("Recent spectate events", _obj)},
        }
    },
    "/api/v1/spectate/status": {
        "get": {
            "tags": ["Debates"],
            "summary": "Get spectate bridge status",
            "description": "Return bridge readiness, subscriber counts, and recent activity metadata for the public spectate surface.",
            "operationId": "getSpectateStatus",
            "responses": {"200": _ok_response("Spectate bridge status", _obj)},
        }
    },
    "/api/v1/spectate/stream": {
        "get": {
            "tags": ["Debates"],
            "summary": "Get spectate stream snapshot",
            "description": "Return a finite SSE snapshot or JSON preview from the buffered spectate stream.",
            "operationId": "getSpectateStream",
            "parameters": [
                {
                    "name": "count",
                    "in": "query",
                    "description": "Maximum number of buffered events to include.",
                    "schema": {"type": "integer", "default": 50, "minimum": 1, "maximum": 500},
                },
                {
                    "name": "debate_id",
                    "in": "query",
                    "description": "Limit results to one debate.",
                    "schema": _str,
                },
                {
                    "name": "pipeline_id",
                    "in": "query",
                    "description": "Limit results to one pipeline.",
                    "schema": _str,
                },
                {
                    "name": "format",
                    "in": "query",
                    "description": "Use `sse` to request a finite SSE snapshot instead of JSON.",
                    "schema": {"type": "string", "enum": ["json", "sse"]},
                },
            ],
            "responses": {"200": _ok_response("Spectate stream snapshot", _obj)},
        }
    },
    "/api/v1/spectate/emit": {
        "post": {
            "tags": ["Debates"],
            "summary": "Inject spectate events",
            "description": "Push one or more events into the spectate bridge for internal demos or tests.",
            "operationId": "createSpectateEmit",
            "responses": {
                "200": _ok_response(
                    "Spectate events emitted",
                    {
                        "type": "object",
                        "properties": {
                            "emitted": _int,
                            "debate_id": _str,
                        },
                    },
                ),
                "503": _ok_response("Bridge unavailable", _obj),
            },
        }
    },
}
