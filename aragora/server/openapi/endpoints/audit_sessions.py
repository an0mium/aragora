"""Audit session endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS, AUTH_REQUIREMENTS

AUDIT_SESSIONS_ENDPOINTS = {
    "/api/v1/audit/sessions": {
        "get": {
            "tags": ["Audit"],
            "summary": "List audit sessions",
            "operationId": "listAuditSessions",
            "description": "List audit sessions with optional filters.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "status", "in": "query", "schema": {"type": "string"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer"}},
                {"name": "offset", "in": "query", "schema": {"type": "integer"}},
            ],
            "responses": {
                "200": _ok_response("Sessions", "StandardSuccessResponse"),
                "500": STANDARD_ERRORS["500"],
            },
        },
        "post": {
            "tags": ["Audit"],
            "summary": "Create audit session",
            "operationId": "createAuditSessions",
            "description": "Create a new audit session.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Session created", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/audit/sessions/{session_id}": {
        "get": {
            "tags": ["Audit"],
            "summary": "Get audit session",
            "operationId": "getAuditSession",
            "description": "Get audit session details.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "session_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Session", "StandardSuccessResponse"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "delete": {
            "tags": ["Audit"],
            "summary": "Delete audit session",
            "operationId": "deleteAuditSession",
            "description": "Delete an audit session.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "session_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Session deleted", "StandardSuccessResponse"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/audit/sessions/{session_id}/start": {
        "post": {
            "tags": ["Audit"],
            "summary": "Start audit",
            "operationId": "createAuditSessionsStart",
            "description": "Start an audit session.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "session_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Audit started", "StandardSuccessResponse"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/audit/sessions/{session_id}/pause": {
        "post": {
            "tags": ["Audit"],
            "summary": "Pause audit",
            "operationId": "createAuditSessionsPause",
            "description": "Pause an audit session.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "session_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Audit paused", "StandardSuccessResponse"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/audit/sessions/{session_id}/resume": {
        "post": {
            "tags": ["Audit"],
            "summary": "Resume audit",
            "operationId": "createAuditSessionsResume",
            "description": "Resume an audit session.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "session_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Audit resumed", "StandardSuccessResponse"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/audit/sessions/{session_id}/cancel": {
        "post": {
            "tags": ["Audit"],
            "summary": "Cancel audit",
            "operationId": "createAuditSessionsCancel",
            "description": "Cancel an audit session.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "session_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Audit cancelled", "StandardSuccessResponse"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/audit/sessions/{session_id}/findings": {
        "get": {
            "tags": ["Audit"],
            "summary": "List findings",
            "operationId": "getAuditSessionsFinding",
            "description": "List audit findings for a session.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "session_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {"name": "audit_type", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Findings", "StandardSuccessResponse"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/audit/sessions/{session_id}/events": {
        "get": {
            "tags": ["Audit"],
            "summary": "Stream audit events",
            "operationId": "getAuditSessionsEvent",
            "description": "Stream audit events via SSE.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "session_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Event stream", "StandardSuccessResponse"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/audit/sessions/{session_id}/intervene": {
        "post": {
            "tags": ["Audit"],
            "summary": "Intervene in audit",
            "operationId": "createAuditSessionsIntervene",
            "description": "Submit a human intervention.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "session_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Intervention recorded", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/audit/sessions/{session_id}/report": {
        "get": {
            "tags": ["Audit"],
            "summary": "Export audit report",
            "operationId": "getAuditSessionsReport",
            "description": "Export audit report for a session.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "session_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {"name": "format", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Report", "StandardSuccessResponse"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
}


__all__ = ["AUDIT_SESSIONS_ENDPOINTS"]
