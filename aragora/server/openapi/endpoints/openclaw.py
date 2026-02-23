"""OpenClaw Gateway endpoint definitions.

Stability: STABLE

Provides OpenAPI specification for OpenClaw gateway endpoints:
- Session management (create, get, list, close)
- Action execution (execute, status, cancel)
- Credential management (store, list, delete, rotate)
- Policy management (get, add, remove rules)
- Approval workflow (list, approve, deny)
- Admin operations (health, metrics, audit, stats)
"""

from typing import Any

from aragora.server.openapi.helpers import STANDARD_ERRORS


def _response(description: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a response with optional inline schema."""
    resp: dict[str, Any] = {"description": description}
    if schema:
        resp["content"] = {"application/json": {"schema": schema}}
    return resp


# =============================================================================
# Reusable Schemas
# =============================================================================

_SESSION_STATUS_ENUM = ["active", "idle", "closing", "closed", "error"]
_ACTION_STATUS_ENUM = ["pending", "running", "completed", "failed", "cancelled", "timeout"]
_CREDENTIAL_TYPE_ENUM = [
    "api_key",
    "oauth_token",
    "password",
    "certificate",
    "ssh_key",
    "service_account",
]

_SESSION_SCHEMA: dict[str, Any] = {
    "type": ["object", "null"],
    "properties": {
        "id": {"type": ["string", "null"], "description": "Unique session identifier"},
        "user_id": {"type": ["string", "null"], "description": "User who owns the session"},
        "tenant_id": {"type": ["string", "null"], "description": "Tenant identifier"},
        "status": {
            "type": ["string", "null"],
            "enum": _SESSION_STATUS_ENUM,
            "description": "Current session status",
        },
        "created_at": {
            "type": ["string", "null"],
            "format": "date-time",
            "description": "Session creation timestamp",
        },
        "updated_at": {
            "type": ["string", "null"],
            "format": "date-time",
            "description": "Last update timestamp",
        },
        "last_activity_at": {
            "type": ["string", "null"],
            "format": "date-time",
            "description": "Last activity timestamp",
        },
        "config": {"type": "object", "description": "Session configuration"},
        "metadata": {"type": "object", "description": "Session metadata"},
    },
}

_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "Unique action identifier"},
        "session_id": {"type": "string", "description": "Associated session ID"},
        "action_type": {"type": "string", "description": "Type of action to execute"},
        "status": {
            "type": "string",
            "enum": _ACTION_STATUS_ENUM,
            "description": "Current action status",
        },
        "input_data": {"type": "object", "description": "Input parameters for the action"},
        "output_data": {
            "type": ["object", "null"],
            "description": "Action output (when completed)",
        },
        "error": {"type": ["string", "null"], "description": "Error message (if failed)"},
        "created_at": {
            "type": "string",
            "format": "date-time",
            "description": "Action creation timestamp",
        },
        "started_at": {
            "type": "string",
            "format": "date-time",
            "description": "Action start timestamp",
        },
        "completed_at": {
            "type": "string",
            "format": "date-time",
            "description": "Action completion timestamp",
        },
        "metadata": {"type": "object", "description": "Action metadata"},
    },
}

_CREDENTIAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "Credential metadata (never includes actual secret values)",
    "properties": {
        "id": {"type": "string", "description": "Unique credential identifier"},
        "name": {"type": "string", "description": "Human-readable credential name"},
        "credential_type": {
            "type": "string",
            "enum": _CREDENTIAL_TYPE_ENUM,
            "description": "Type of credential",
        },
        "user_id": {"type": "string", "description": "User who owns the credential"},
        "tenant_id": {"type": ["string", "null"], "description": "Tenant identifier"},
        "created_at": {
            "type": "string",
            "format": "date-time",
            "description": "Creation timestamp",
        },
        "updated_at": {
            "type": "string",
            "format": "date-time",
            "description": "Last update timestamp",
        },
        "last_rotated_at": {
            "type": "string",
            "format": "date-time",
            "description": "Last rotation timestamp",
        },
        "expires_at": {
            "type": "string",
            "format": "date-time",
            "description": "Expiration timestamp",
        },
        "metadata": {"type": "object", "description": "Credential metadata"},
    },
}

_POLICY_RULE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Unique rule name"},
        "description": {"type": "string", "description": "Rule description"},
        "action_pattern": {"type": "string", "description": "Action pattern to match"},
        "conditions": {
            "type": "array",
            "items": {"type": "object"},
            "description": "Conditions for rule application",
        },
        "effect": {
            "type": "string",
            "enum": ["allow", "deny", "require_approval"],
            "description": "Rule effect",
        },
        "priority": {"type": "integer", "description": "Rule evaluation priority"},
        "enabled": {"type": "boolean", "description": "Whether the rule is active"},
    },
}

_APPROVAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "Unique approval identifier"},
        "action_id": {"type": "string", "description": "Associated action ID"},
        "session_id": {"type": "string", "description": "Associated session ID"},
        "user_id": {"type": "string", "description": "User who requested the action"},
        "status": {
            "type": "string",
            "enum": ["pending", "approved", "denied"],
            "description": "Approval status",
        },
        "action_type": {"type": "string", "description": "Type of action requiring approval"},
        "action_data": {"type": "object", "description": "Action input data for review"},
        "requested_at": {
            "type": "string",
            "format": "date-time",
            "description": "Request timestamp",
        },
        "decided_at": {
            "type": "string",
            "format": "date-time",
            "description": "Decision timestamp",
        },
        "decided_by": {
            "type": ["string", "null"],
            "description": "User who made the decision",
        },
        "reason": {"type": ["string", "null"], "description": "Decision reason"},
    },
}

_AUDIT_ENTRY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "Unique audit entry ID"},
        "timestamp": {"type": "string", "format": "date-time", "description": "Event timestamp"},
        "event_type": {"type": "string", "description": "Type of event"},
        "user_id": {"type": "string", "description": "User who triggered the event"},
        "session_id": {"type": ["string", "null"], "description": "Associated session"},
        "action_id": {"type": ["string", "null"], "description": "Associated action"},
        "details": {"type": "object", "description": "Event details"},
    },
}

# =============================================================================
# Path Parameters
# =============================================================================

_SESSION_ID_PARAM: dict[str, Any] = {
    "name": "session_id",
    "in": "path",
    "required": True,
    "description": "Unique session identifier",
    "schema": {"type": "string"},
}

_ACTION_ID_PARAM: dict[str, Any] = {
    "name": "action_id",
    "in": "path",
    "required": True,
    "description": "Unique action identifier",
    "schema": {"type": "string"},
}

_CREDENTIAL_ID_PARAM: dict[str, Any] = {
    "name": "credential_id",
    "in": "path",
    "required": True,
    "description": "Unique credential identifier",
    "schema": {"type": "string"},
}

_RULE_NAME_PARAM: dict[str, Any] = {
    "name": "rule_name",
    "in": "path",
    "required": True,
    "description": "Policy rule name",
    "schema": {"type": "string"},
}

_APPROVAL_ID_PARAM: dict[str, Any] = {
    "name": "approval_id",
    "in": "path",
    "required": True,
    "description": "Unique approval identifier",
    "schema": {"type": "string"},
}

# =============================================================================
# OpenClaw Endpoint Definitions
# =============================================================================

OPENCLAW_ENDPOINTS: dict[str, Any] = {
    # =========================================================================
    # Session Management
    # =========================================================================
    "/api/v1/openclaw/sessions": {
        "get": {
            "tags": ["OpenClaw"],
            "summary": "List OpenClaw sessions",
            "description": "List all OpenClaw sessions for the current user/tenant.",
            "operationId": "listOpenclawSessions",
            "parameters": [
                {
                    "name": "status",
                    "in": "query",
                    "description": "Filter by session status",
                    "schema": {"type": "string", "enum": _SESSION_STATUS_ENUM},
                },
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of sessions to return",
                    "schema": {"type": "integer", "default": 50, "maximum": 100},
                },
                {
                    "name": "offset",
                    "in": "query",
                    "description": "Number of sessions to skip",
                    "schema": {"type": "integer", "default": 0},
                },
            ],
            "responses": {
                "200": _response(
                    "List of sessions",
                    {
                        "type": "object",
                        "properties": {
                            "sessions": {"type": "array", "items": _SESSION_SCHEMA},
                            "total": {"type": "integer"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "post": {
            "tags": ["OpenClaw"],
            "summary": "Create OpenClaw session",
            "description": "Create a new OpenClaw session for executing actions.",
            "operationId": "createOpenclawSession",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "config": {
                                    "type": "object",
                                    "description": "Session configuration",
                                },
                                "metadata": {
                                    "type": "object",
                                    "description": "Session metadata",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "201": _response(
                    "Session created",
                    {
                        "type": "object",
                        "properties": {
                            "session": _SESSION_SCHEMA,
                            "message": {"type": "string"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/openclaw/sessions/{session_id}": {
        "get": {
            "tags": ["OpenClaw"],
            "summary": "Get OpenClaw session",
            "description": "Get details of a specific OpenClaw session.",
            "operationId": "getOpenclawSession",
            "parameters": [_SESSION_ID_PARAM],
            "responses": {
                "200": _response(
                    "Session details",
                    {
                        "type": "object",
                        "properties": {"session": _SESSION_SCHEMA},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "delete": {
            "tags": ["OpenClaw"],
            "summary": "Close OpenClaw session",
            "description": "Close an OpenClaw session. Running actions will be cancelled.",
            "operationId": "closeOpenclawSession",
            "parameters": [_SESSION_ID_PARAM],
            "responses": {
                "200": _response(
                    "Session closed",
                    {
                        "type": "object",
                        "properties": {
                            "session": _SESSION_SCHEMA,
                            "message": {"type": "string"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/openclaw/sessions/{session_id}/end": {
        "post": {
            "tags": ["OpenClaw"],
            "summary": "End OpenClaw session",
            "description": "Gracefully end an OpenClaw session, waiting for running actions.",
            "operationId": "endOpenclawSession",
            "parameters": [_SESSION_ID_PARAM],
            "responses": {
                "200": _response(
                    "Session ended",
                    {
                        "type": "object",
                        "properties": {
                            "session": _SESSION_SCHEMA,
                            "message": {"type": "string"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    # =========================================================================
    # Action Management
    # =========================================================================
    "/api/v1/openclaw/actions": {
        "post": {
            "tags": ["OpenClaw"],
            "summary": "Execute OpenClaw action",
            "description": "Execute an action within an OpenClaw session.",
            "operationId": "executeOpenclawAction",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["session_id", "action_type"],
                            "properties": {
                                "session_id": {
                                    "type": "string",
                                    "description": "Session to execute action in",
                                },
                                "action_type": {
                                    "type": "string",
                                    "description": "Type of action to execute",
                                },
                                "input_data": {
                                    "type": "object",
                                    "description": "Input parameters for the action",
                                },
                                "metadata": {
                                    "type": "object",
                                    "description": "Action metadata",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "202": _response(
                    "Action queued for execution",
                    {
                        "type": "object",
                        "properties": {
                            "action": _ACTION_SCHEMA,
                            "message": {"type": "string"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/openclaw/actions/{action_id}": {
        "get": {
            "tags": ["OpenClaw"],
            "summary": "Get action status",
            "description": "Get the current status and details of an OpenClaw action.",
            "operationId": "getOpenclawAction",
            "parameters": [_ACTION_ID_PARAM],
            "responses": {
                "200": _response(
                    "Action details",
                    {
                        "type": "object",
                        "properties": {"action": _ACTION_SCHEMA},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/openclaw/actions/{action_id}/cancel": {
        "post": {
            "tags": ["OpenClaw"],
            "summary": "Cancel OpenClaw action",
            "description": "Cancel a running or pending OpenClaw action.",
            "operationId": "cancelOpenclawAction",
            "parameters": [_ACTION_ID_PARAM],
            "responses": {
                "200": _response(
                    "Action cancelled",
                    {
                        "type": "object",
                        "properties": {
                            "action": _ACTION_SCHEMA,
                            "message": {"type": "string"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    # =========================================================================
    # Credential Management
    # =========================================================================
    "/api/v1/openclaw/credentials": {
        "get": {
            "tags": ["OpenClaw"],
            "summary": "List credentials",
            "description": "List credential metadata (not secret values).",
            "operationId": "listOpenclawCredentials",
            "parameters": [
                {
                    "name": "credential_type",
                    "in": "query",
                    "description": "Filter by credential type",
                    "schema": {"type": "string", "enum": _CREDENTIAL_TYPE_ENUM},
                },
            ],
            "responses": {
                "200": _response(
                    "List of credential metadata",
                    {
                        "type": "object",
                        "properties": {
                            "credentials": {"type": "array", "items": _CREDENTIAL_SCHEMA},
                            "total": {"type": "integer"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "post": {
            "tags": ["OpenClaw"],
            "summary": "Store credential",
            "description": "Securely store a new credential for OpenClaw actions.",
            "operationId": "storeOpenclawCredential",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["name", "credential_type", "value"],
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Human-readable credential name",
                                },
                                "credential_type": {
                                    "type": "string",
                                    "enum": _CREDENTIAL_TYPE_ENUM,
                                    "description": "Type of credential",
                                },
                                "value": {
                                    "type": "string",
                                    "description": "The credential secret value",
                                },
                                "expires_at": {
                                    "type": "string",
                                    "format": "date-time",
                                    "description": "Optional expiration timestamp",
                                },
                                "metadata": {
                                    "type": "object",
                                    "description": "Additional metadata",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "201": _response(
                    "Credential stored",
                    {
                        "type": "object",
                        "properties": {
                            "credential": _CREDENTIAL_SCHEMA,
                            "message": {"type": "string"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/openclaw/credentials/{credential_id}": {
        "delete": {
            "tags": ["OpenClaw"],
            "summary": "Delete credential",
            "description": "Delete a stored credential.",
            "operationId": "deleteOpenclawCredential",
            "parameters": [_CREDENTIAL_ID_PARAM],
            "responses": {
                "200": _response(
                    "Credential deleted",
                    {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/openclaw/credentials/{credential_id}/rotate": {
        "post": {
            "tags": ["OpenClaw"],
            "summary": "Rotate credential",
            "description": "Rotate a credential with a new value.",
            "operationId": "rotateOpenclawCredential",
            "parameters": [_CREDENTIAL_ID_PARAM],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["new_value"],
                            "properties": {
                                "new_value": {
                                    "type": "string",
                                    "description": "New credential secret value",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _response(
                    "Credential rotated",
                    {
                        "type": "object",
                        "properties": {
                            "credential": _CREDENTIAL_SCHEMA,
                            "message": {"type": "string"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    # =========================================================================
    # Policy Management
    # =========================================================================
    "/api/v1/openclaw/policy/rules": {
        "get": {
            "tags": ["OpenClaw"],
            "summary": "Get policy rules",
            "description": "List all OpenClaw policy rules.",
            "operationId": "getOpenclawPolicyRules",
            "parameters": [
                {
                    "name": "enabled",
                    "in": "query",
                    "description": "Filter by enabled status",
                    "schema": {"type": "boolean"},
                },
            ],
            "responses": {
                "200": _response(
                    "List of policy rules",
                    {
                        "type": "object",
                        "properties": {
                            "rules": {"type": "array", "items": _POLICY_RULE_SCHEMA},
                            "total": {"type": "integer"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "post": {
            "tags": ["OpenClaw"],
            "summary": "Add policy rule",
            "description": "Add a new OpenClaw policy rule.",
            "operationId": "addOpenclawPolicyRule",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["name", "action_pattern", "effect"],
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Unique rule name",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Rule description",
                                },
                                "action_pattern": {
                                    "type": "string",
                                    "description": "Action pattern to match",
                                },
                                "conditions": {
                                    "type": "array",
                                    "items": {"type": "object"},
                                    "description": "Conditions for rule application",
                                },
                                "effect": {
                                    "type": "string",
                                    "enum": ["allow", "deny", "require_approval"],
                                    "description": "Rule effect",
                                },
                                "priority": {
                                    "type": "integer",
                                    "description": "Rule evaluation priority",
                                },
                                "enabled": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Whether the rule is active",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "201": _response(
                    "Policy rule created",
                    {
                        "type": "object",
                        "properties": {
                            "rule": _POLICY_RULE_SCHEMA,
                            "message": {"type": "string"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/openclaw/policy/rules/{rule_name}": {
        "delete": {
            "tags": ["OpenClaw"],
            "summary": "Remove policy rule",
            "description": "Remove an OpenClaw policy rule.",
            "operationId": "removeOpenclawPolicyRule",
            "parameters": [_RULE_NAME_PARAM],
            "responses": {
                "200": _response(
                    "Policy rule removed",
                    {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    # =========================================================================
    # Approval Workflow
    # =========================================================================
    "/api/v1/openclaw/approvals": {
        "get": {
            "tags": ["OpenClaw"],
            "summary": "List pending approvals",
            "description": "List actions requiring approval.",
            "operationId": "listOpenclawApprovals",
            "parameters": [
                {
                    "name": "status",
                    "in": "query",
                    "description": "Filter by approval status",
                    "schema": {"type": "string", "enum": ["pending", "approved", "denied"]},
                },
            ],
            "responses": {
                "200": _response(
                    "List of approvals",
                    {
                        "type": "object",
                        "properties": {
                            "approvals": {"type": "array", "items": _APPROVAL_SCHEMA},
                            "total": {"type": "integer"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/openclaw/approvals/{approval_id}/approve": {
        "post": {
            "tags": ["OpenClaw"],
            "summary": "Approve action",
            "description": "Approve a pending action request.",
            "operationId": "approveOpenclawAction",
            "parameters": [_APPROVAL_ID_PARAM],
            "requestBody": {
                "required": False,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "reason": {
                                    "type": "string",
                                    "description": "Optional approval reason",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _response(
                    "Action approved",
                    {
                        "type": "object",
                        "properties": {
                            "approval": _APPROVAL_SCHEMA,
                            "message": {"type": "string"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/openclaw/approvals/{approval_id}/deny": {
        "post": {
            "tags": ["OpenClaw"],
            "summary": "Deny action",
            "description": "Deny a pending action request.",
            "operationId": "denyOpenclawAction",
            "parameters": [_APPROVAL_ID_PARAM],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["reason"],
                            "properties": {
                                "reason": {
                                    "type": "string",
                                    "description": "Reason for denial",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _response(
                    "Action denied",
                    {
                        "type": "object",
                        "properties": {
                            "approval": _APPROVAL_SCHEMA,
                            "message": {"type": "string"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    # =========================================================================
    # Admin Endpoints
    # =========================================================================
    "/api/v1/openclaw/health": {
        "get": {
            "tags": ["OpenClaw"],
            "summary": "Gateway health check",
            "description": "Check the health status of the OpenClaw gateway.",
            "operationId": "getOpenclawHealth",
            "responses": {
                "200": _response(
                    "Gateway health status",
                    {
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["healthy", "degraded", "unhealthy"],
                            },
                            "components": {
                                "type": "object",
                                "description": "Individual component health statuses",
                            },
                            "timestamp": {"type": "string", "format": "date-time"},
                        },
                    },
                ),
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/openclaw/metrics": {
        "get": {
            "tags": ["OpenClaw"],
            "summary": "Gateway metrics",
            "description": "Get OpenClaw gateway performance metrics.",
            "operationId": "getOpenclawMetrics",
            "responses": {
                "200": _response(
                    "Gateway metrics",
                    {
                        "type": "object",
                        "properties": {
                            "sessions": {
                                "type": "object",
                                "properties": {
                                    "active": {"type": "integer"},
                                    "total": {"type": "integer"},
                                },
                            },
                            "actions": {
                                "type": "object",
                                "properties": {
                                    "pending": {"type": "integer"},
                                    "running": {"type": "integer"},
                                    "completed": {"type": "integer"},
                                    "failed": {"type": "integer"},
                                },
                            },
                            "circuit_breaker": {
                                "type": "object",
                                "properties": {
                                    "state": {"type": "string"},
                                    "failure_count": {"type": "integer"},
                                },
                            },
                            "timestamp": {"type": "string", "format": "date-time"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/openclaw/audit": {
        "get": {
            "tags": ["OpenClaw"],
            "summary": "Get audit log",
            "description": "Get OpenClaw gateway audit log entries.",
            "operationId": "getOpenclawAudit",
            "parameters": [
                {
                    "name": "event_type",
                    "in": "query",
                    "description": "Filter by event type",
                    "schema": {"type": "string"},
                },
                {
                    "name": "user_id",
                    "in": "query",
                    "description": "Filter by user ID",
                    "schema": {"type": "string"},
                },
                {
                    "name": "session_id",
                    "in": "query",
                    "description": "Filter by session ID",
                    "schema": {"type": "string"},
                },
                {
                    "name": "start_time",
                    "in": "query",
                    "description": "Filter entries after this timestamp",
                    "schema": {"type": "string", "format": "date-time"},
                },
                {
                    "name": "end_time",
                    "in": "query",
                    "description": "Filter entries before this timestamp",
                    "schema": {"type": "string", "format": "date-time"},
                },
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of entries to return",
                    "schema": {"type": "integer", "default": 100, "maximum": 1000},
                },
            ],
            "responses": {
                "200": _response(
                    "Audit log entries",
                    {
                        "type": "object",
                        "properties": {
                            "entries": {"type": "array", "items": _AUDIT_ENTRY_SCHEMA},
                            "total": {"type": "integer"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/openclaw/stats": {
        "get": {
            "tags": ["OpenClaw"],
            "summary": "Gateway statistics",
            "description": "Get comprehensive OpenClaw gateway statistics.",
            "operationId": "getOpenclawStats",
            "responses": {
                "200": _response(
                    "Gateway statistics",
                    {
                        "type": "object",
                        "properties": {
                            "sessions_created": {"type": "integer"},
                            "actions_executed": {"type": "integer"},
                            "actions_succeeded": {"type": "integer"},
                            "actions_failed": {"type": "integer"},
                            "approvals_pending": {"type": "integer"},
                            "credentials_stored": {"type": "integer"},
                            "policy_rules_active": {"type": "integer"},
                            "uptime_seconds": {"type": "number"},
                            "timestamp": {"type": "string", "format": "date-time"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
}

__all__ = ["OPENCLAW_ENDPOINTS"]
