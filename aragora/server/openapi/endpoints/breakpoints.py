"""Breakpoint endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS

_BREAKPOINTS_UNAVAILABLE = {
    "description": "Breakpoints module not available",
    "content": {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "error": {"type": "string"},
                },
            }
        }
    },
}

_BREAKPOINT_SNAPSHOT_SCHEMA = {
    "type": ["object", "null"],
    "properties": {
        "debate_id": {"type": "string", "description": "Debate identifier"},
        "round_num": {"type": "integer", "description": "Round number at the breakpoint"},
        "task": {"type": "string", "description": "Debate task under review"},
        "confidence": {
            "type": ["number", "null"],
            "description": "Confidence score at the breakpoint",
        },
        "agents": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Agents participating when the breakpoint triggered",
        },
    },
}

_BREAKPOINT_SCHEMA = {
    "type": "object",
    "properties": {
        "breakpoint_id": {"type": "string", "description": "Breakpoint identifier"},
        "trigger": {"type": "string", "description": "Breakpoint trigger reason"},
        "message": {"type": "string", "description": "Human-readable breakpoint summary"},
        "created_at": {"type": "string", "description": "Creation timestamp"},
        "timeout_minutes": {
            "type": "integer",
            "description": "Minutes before the breakpoint times out",
        },
        "status": {
            "type": "string",
            "description": "Current breakpoint status",
        },
        "resolved_at": {
            "type": ["string", "null"],
            "description": "Resolution timestamp when available",
        },
        "snapshot": _BREAKPOINT_SNAPSHOT_SCHEMA,
    },
}

BREAKPOINT_ENDPOINTS = {
    "/api/v1/breakpoints": {
        "get": {
            "tags": ["Checkpoints"],
            "summary": "List pending breakpoints",
            "operationId": "listBreakpoints",
            "description": "List pending human-in-the-loop breakpoints awaiting review.",
            "responses": {
                "200": _ok_response(
                    "Pending breakpoints",
                    {
                        "type": "object",
                        "properties": {
                            "breakpoints": {
                                "type": "array",
                                "items": _BREAKPOINT_SCHEMA,
                            },
                            "count": {
                                "type": "integer",
                                "description": "Number of pending breakpoints",
                            },
                        },
                    },
                ),
                "503": _BREAKPOINTS_UNAVAILABLE,
            },
        },
    },
    "/api/v1/breakpoints/pending": {
        "get": {
            "tags": ["Checkpoints"],
            "summary": "List pending breakpoints (alias)",
            "operationId": "listPendingBreakpoints",
            "description": "Canonical alias for listing pending human-in-the-loop breakpoints.",
            "responses": {
                "200": _ok_response(
                    "Pending breakpoints",
                    {
                        "type": "object",
                        "properties": {
                            "breakpoints": {
                                "type": "array",
                                "items": _BREAKPOINT_SCHEMA,
                            },
                            "count": {
                                "type": "integer",
                                "description": "Number of pending breakpoints",
                            },
                        },
                    },
                ),
                "503": _BREAKPOINTS_UNAVAILABLE,
            },
        },
    },
    "/api/v1/breakpoints/{breakpoint_id}/status": {
        "get": {
            "tags": ["Checkpoints"],
            "summary": "Get breakpoint status",
            "operationId": "getBreakpointStatus",
            "description": "Return the current status and snapshot for a specific breakpoint.",
            "parameters": [
                {
                    "name": "breakpoint_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Breakpoint identifier",
                }
            ],
            "responses": {
                "200": _ok_response("Breakpoint status", _BREAKPOINT_SCHEMA),
                "404": STANDARD_ERRORS["404"],
                "503": _BREAKPOINTS_UNAVAILABLE,
            },
        },
    },
    "/api/v1/breakpoints/{breakpoint_id}/resolve": {
        "post": {
            "tags": ["Checkpoints"],
            "summary": "Resolve a breakpoint",
            "operationId": "resolveBreakpoint",
            "description": "Resolve a pending breakpoint with human guidance.",
            "parameters": [
                {
                    "name": "breakpoint_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Breakpoint identifier",
                }
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["action"],
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["continue", "abort", "redirect", "inject"],
                                    "description": "Human-selected resolution action",
                                },
                                "message": {
                                    "type": "string",
                                    "description": "Reasoning or note for the resolution",
                                },
                                "reviewer_id": {
                                    "type": "string",
                                    "description": "Reviewer identifier for audit trails",
                                },
                                "redirect_task": {
                                    "type": "string",
                                    "description": "Replacement task when action is redirect",
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response(
                    "Breakpoint resolved",
                    {
                        "type": "object",
                        "properties": {
                            "breakpoint_id": {"type": "string"},
                            "status": {"type": "string", "enum": ["resolved"]},
                            "action": {"type": "string"},
                            "message": {"type": "string"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
                "503": _BREAKPOINTS_UNAVAILABLE,
            },
        },
    },
}

__all__ = ["BREAKPOINT_ENDPOINTS"]
