"""Knowledge Mound governance endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS, AUTH_REQUIREMENTS


KNOWLEDGE_MOUND_ENDPOINTS = {
    "/api/v1/knowledge/mound/governance/roles": {
        "post": {
            "tags": ["Knowledge Mound"],
            "summary": "Create role",
            "description": "Create a Knowledge Mound governance role.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Role created", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/knowledge/mound/governance/roles/assign": {
        "post": {
            "tags": ["Knowledge Mound"],
            "summary": "Assign role",
            "description": "Assign a governance role to a user.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Role assigned", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/knowledge/mound/governance/roles/revoke": {
        "post": {
            "tags": ["Knowledge Mound"],
            "summary": "Revoke role",
            "description": "Revoke a governance role from a user.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Role revoked", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/knowledge/mound/governance/permissions/{user_id}": {
        "get": {
            "tags": ["Knowledge Mound"],
            "summary": "Get permissions",
            "description": "Get governance permissions for a user.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "user_id", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Permissions", "StandardSuccessResponse"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/knowledge/mound/governance/permissions/check": {
        "post": {
            "tags": ["Knowledge Mound"],
            "summary": "Check permissions",
            "description": "Check a permission against governance rules.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Permission check", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/knowledge/mound/governance/audit": {
        "get": {
            "tags": ["Knowledge Mound"],
            "summary": "Audit trail",
            "description": "Query governance audit trail.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "limit", "in": "query", "schema": {"type": "integer"}},
                {"name": "offset", "in": "query", "schema": {"type": "integer"}},
            ],
            "responses": {
                "200": _ok_response("Audit trail", "StandardSuccessResponse"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/knowledge/mound/governance/audit/user/{user_id}": {
        "get": {
            "tags": ["Knowledge Mound"],
            "summary": "User audit trail",
            "description": "Get governance activity for a user.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "user_id", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("User audit", "StandardSuccessResponse"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/knowledge/mound/governance/stats": {
        "get": {
            "tags": ["Knowledge Mound"],
            "summary": "Governance stats",
            "description": "Get governance stats.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "responses": {
                "200": _ok_response("Stats", "StandardSuccessResponse"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
}


__all__ = ["KNOWLEDGE_MOUND_ENDPOINTS"]
