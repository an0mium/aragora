"""Workspace and privacy endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS

WORKSPACE_ENDPOINTS = {
    "/api/workspaces": {
        "get": {
            "tags": ["Workspace"],
            "summary": "List workspaces",
            "description": "Get list of workspaces the authenticated user has access to.",
            "security": [{"bearerAuth": []}],
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 50, "maximum": 200},
                },
                {
                    "name": "offset",
                    "in": "query",
                    "schema": {"type": "integer", "default": 0},
                },
            ],
            "responses": {
                "200": _ok_response("List of workspaces", "WorkspaceList"),
                "401": STANDARD_ERRORS["401"],
            },
        },
        "post": {
            "tags": ["Workspace"],
            "summary": "Create workspace",
            "description": "Create a new isolated workspace for data segregation.",
            "security": [{"bearerAuth": []}],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["name"],
                            "properties": {
                                "name": {"type": "string", "description": "Workspace name"},
                                "description": {"type": "string"},
                                "sensitivity_default": {
                                    "type": "string",
                                    "enum": ["public", "internal", "confidential", "restricted"],
                                    "default": "internal",
                                },
                                "retention_days": {"type": "integer", "default": 365},
                            },
                        },
                    },
                },
            },
            "responses": {
                "201": _ok_response("Workspace created", "Workspace"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
            },
        },
    },
    "/api/workspaces/{workspace_id}": {
        "get": {
            "tags": ["Workspace"],
            "summary": "Get workspace details",
            "description": "Get detailed information about a specific workspace.",
            "security": [{"bearerAuth": []}],
            "parameters": [
                {
                    "name": "workspace_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Workspace details", "Workspace"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
            },
        },
        "delete": {
            "tags": ["Workspace"],
            "summary": "Delete workspace",
            "description": "Delete a workspace and all associated data. Requires admin permissions.",
            "security": [{"bearerAuth": []}],
            "parameters": [
                {
                    "name": "workspace_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Workspace deleted"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/workspaces/{workspace_id}/members": {
        "post": {
            "tags": ["Workspace"],
            "summary": "Add workspace member",
            "description": "Add a user to the workspace with specified permissions.",
            "security": [{"bearerAuth": []}],
            "parameters": [
                {
                    "name": "workspace_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["user_id", "permission"],
                            "properties": {
                                "user_id": {"type": "string"},
                                "permission": {
                                    "type": "string",
                                    "enum": ["read", "write", "admin"],
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _ok_response("Member added"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
            },
        },
    },
    "/api/workspaces/profiles": {
        "get": {
            "tags": ["Workspace"],
            "summary": "List RBAC profiles",
            "description": """Get available RBAC profiles for workspace configuration.

**Profiles:**
- **lite**: 3 roles (owner, admin, member) - ideal for SME workspaces
- **standard**: 5 roles (adds analyst, viewer) - for growing teams
- **enterprise**: 8 roles - full governance with compliance roles

**Recommendation:** Use 'lite' for most workspaces; upgrade as needed.""",
            "operationId": "listWorkspaceProfiles",
            "security": [{"bearerAuth": []}],
            "responses": {
                "200": _ok_response("Available RBAC profiles with role details"),
                "401": STANDARD_ERRORS["401"],
            },
        },
    },
    "/api/workspaces/{workspace_id}/roles": {
        "get": {
            "tags": ["Workspace"],
            "summary": "Get workspace roles",
            "description": """Get available roles for a workspace based on its RBAC profile.

**Response includes:**
- Roles available in the workspace's profile
- Which roles the current user can assign
- Current user's role in the workspace

**Role assignment rules:**
- Owners can assign all roles except owner
- Admins can assign member, analyst, viewer
- Members cannot assign roles""",
            "operationId": "getWorkspaceRoles",
            "security": [{"bearerAuth": []}],
            "parameters": [
                {
                    "name": "workspace_id",
                    "in": "path",
                    "required": True,
                    "description": "Workspace ID",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Workspace roles with assignment permissions"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/workspaces/{workspace_id}/members/{user_id}/role": {
        "put": {
            "tags": ["Workspace"],
            "summary": "Update member role",
            "description": """Update a workspace member's role.

**Role hierarchy:** owner > admin > member

**Restrictions:**
- Cannot remove the last owner
- Can only assign roles within your permission level
- Role changes are audit logged

**Lite profile roles:**
- `owner`: Full workspace control including billing
- `admin`: Manage users and debates, no billing
- `member`: Create and run debates""",
            "operationId": "updateWorkspaceMemberRole",
            "security": [{"bearerAuth": []}],
            "parameters": [
                {
                    "name": "workspace_id",
                    "in": "path",
                    "required": True,
                    "description": "Workspace ID",
                    "schema": {"type": "string"},
                },
                {
                    "name": "user_id",
                    "in": "path",
                    "required": True,
                    "description": "User ID of the member",
                    "schema": {"type": "string"},
                },
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["role"],
                            "properties": {
                                "role": {
                                    "type": "string",
                                    "description": "New role for the member",
                                    "enum": ["owner", "admin", "member", "analyst", "viewer"],
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _ok_response("Role updated successfully"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/retention/policies": {
        "get": {
            "tags": ["Retention"],
            "summary": "List retention policies",
            "description": "Get all configured data retention policies.",
            "security": [{"bearerAuth": []}],
            "responses": {
                "200": _ok_response("List of retention policies", "RetentionPolicyList"),
                "401": STANDARD_ERRORS["401"],
            },
        },
        "post": {
            "tags": ["Retention"],
            "summary": "Create retention policy",
            "description": "Create a new data retention policy for automated data lifecycle management.",
            "security": [{"bearerAuth": []}],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["name", "retention_days", "action"],
                            "properties": {
                                "name": {"type": "string"},
                                "retention_days": {"type": "integer", "minimum": 1},
                                "action": {
                                    "type": "string",
                                    "enum": ["delete", "archive", "anonymize"],
                                },
                                "data_types": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "workspace_id": {"type": "string"},
                            },
                        },
                    },
                },
            },
            "responses": {
                "201": _ok_response("Policy created", "RetentionPolicy"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
            },
        },
    },
    "/api/retention/policies/{policy_id}/execute": {
        "post": {
            "tags": ["Retention"],
            "summary": "Execute retention policy",
            "description": "Manually execute a retention policy to process affected data.",
            "security": [{"bearerAuth": []}],
            "parameters": [
                {
                    "name": "policy_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Execution result with affected items count"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/retention/expiring": {
        "get": {
            "tags": ["Retention"],
            "summary": "Get expiring items",
            "description": "Get list of items approaching their retention expiration date.",
            "security": [{"bearerAuth": []}],
            "parameters": [
                {
                    "name": "days",
                    "in": "query",
                    "description": "Days until expiration to include",
                    "schema": {"type": "integer", "default": 30},
                },
            ],
            "responses": {
                "200": _ok_response("List of expiring items"),
                "401": STANDARD_ERRORS["401"],
            },
        },
    },
    "/api/classify": {
        "post": {
            "tags": ["Classification"],
            "summary": "Classify content sensitivity",
            "description": "Analyze content and determine its sensitivity classification level.",
            "security": [{"bearerAuth": []}],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["content"],
                            "properties": {
                                "content": {"type": "string", "description": "Content to classify"},
                                "context": {"type": "string", "description": "Additional context"},
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _ok_response("Classification result with level and confidence"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
            },
        },
    },
    "/api/classify/policy/{level}": {
        "get": {
            "tags": ["Classification"],
            "summary": "Get classification policy",
            "description": "Get handling policy for a specific sensitivity level.",
            "security": [{"bearerAuth": []}],
            "parameters": [
                {
                    "name": "level",
                    "in": "path",
                    "required": True,
                    "schema": {
                        "type": "string",
                        "enum": ["public", "internal", "confidential", "restricted"],
                    },
                },
            ],
            "responses": {
                "200": _ok_response("Policy details for the sensitivity level"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
            },
        },
    },
    "/api/audit/entries": {
        "get": {
            "tags": ["Audit"],
            "summary": "Query audit entries",
            "description": "Search and filter privacy audit log entries. SOC 2 Control: CC6.1",
            "security": [{"bearerAuth": []}],
            "parameters": [
                {"name": "actor_id", "in": "query", "schema": {"type": "string"}},
                {"name": "resource_id", "in": "query", "schema": {"type": "string"}},
                {"name": "action", "in": "query", "schema": {"type": "string"}},
                {
                    "name": "outcome",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["success", "denied", "error"]},
                },
                {
                    "name": "start_time",
                    "in": "query",
                    "schema": {"type": "string", "format": "date-time"},
                },
                {
                    "name": "end_time",
                    "in": "query",
                    "schema": {"type": "string", "format": "date-time"},
                },
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 100}},
            ],
            "responses": {
                "200": _ok_response("Filtered audit entries"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
            },
        },
    },
    "/api/audit/report": {
        "get": {
            "tags": ["Audit"],
            "summary": "Generate compliance report",
            "description": "Generate a compliance report for a specified time period. SOC 2 Control: CC6.3",
            "security": [{"bearerAuth": []}],
            "parameters": [
                {
                    "name": "start_date",
                    "in": "query",
                    "required": True,
                    "schema": {"type": "string", "format": "date"},
                },
                {
                    "name": "end_date",
                    "in": "query",
                    "required": True,
                    "schema": {"type": "string", "format": "date"},
                },
                {
                    "name": "format",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["json", "csv", "pdf"], "default": "json"},
                },
            ],
            "responses": {
                "200": _ok_response("Compliance report"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
            },
        },
    },
    "/api/audit/verify": {
        "get": {
            "tags": ["Audit"],
            "summary": "Verify audit log integrity",
            "description": "Verify the cryptographic integrity of audit log entries to detect tampering.",
            "security": [{"bearerAuth": []}],
            "parameters": [
                {
                    "name": "start_sequence",
                    "in": "query",
                    "schema": {"type": "integer"},
                },
                {
                    "name": "end_sequence",
                    "in": "query",
                    "schema": {"type": "integer"},
                },
            ],
            "responses": {
                "200": _ok_response("Verification result with integrity status"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
            },
        },
    },
}
