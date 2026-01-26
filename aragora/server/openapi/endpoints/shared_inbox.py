"""Shared inbox endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS, AUTH_REQUIREMENTS


INBOX_ENDPOINTS = {
    "/api/v1/inbox/shared": {
        "post": {
            "tags": ["Inbox"],
            "summary": "Create shared inbox",
            "operationId": "createInboxShared",
            "description": "Create a shared inbox for collaborative triage.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/SharedInboxCreateRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response("Inbox created", "SharedInboxResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "get": {
            "tags": ["Inbox"],
            "summary": "List shared inboxes",
            "operationId": "listInboxShared",
            "description": "List shared inboxes for a workspace.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
                {"name": "user_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Inbox list", "SharedInboxListResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/inbox/shared/{id}": {
        "get": {
            "tags": ["Inbox"],
            "summary": "Get shared inbox",
            "operationId": "getInboxShared",
            "description": "Fetch shared inbox details.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Inbox detail", "SharedInboxResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/inbox/shared/{id}/messages": {
        "get": {
            "tags": ["Inbox"],
            "summary": "List inbox messages",
            "operationId": "getInboxSharedMessage",
            "description": "List messages for a shared inbox.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "status", "in": "query", "schema": {"type": "string"}},
                {"name": "assigned_to", "in": "query", "schema": {"type": "string"}},
                {"name": "tag", "in": "query", "schema": {"type": "string"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer"}},
                {"name": "offset", "in": "query", "schema": {"type": "integer"}},
            ],
            "responses": {
                "200": _ok_response("Inbox messages", "SharedInboxMessageListResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/inbox/shared/{id}/messages/{msg_id}/assign": {
        "post": {
            "tags": ["Inbox"],
            "summary": "Assign message",
            "operationId": "createInboxSharedMessagesAssign",
            "description": "Assign a shared inbox message to a team member.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "msg_id", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/SharedInboxAssignRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response("Message assigned", "SharedInboxMessageResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/inbox/shared/{id}/messages/{msg_id}/status": {
        "post": {
            "tags": ["Inbox"],
            "summary": "Update message status",
            "operationId": "createInboxSharedMessagesStatu",
            "description": "Update shared inbox message status.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "msg_id", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/SharedInboxStatusRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response("Message updated", "SharedInboxMessageResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/inbox/shared/{id}/messages/{msg_id}/tag": {
        "post": {
            "tags": ["Inbox"],
            "summary": "Add message tag",
            "operationId": "createInboxSharedMessagesTag",
            "description": "Add a tag to a shared inbox message.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "msg_id", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/SharedInboxTagRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response("Tag added", "SharedInboxMessageResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/inbox/routing/rules": {
        "post": {
            "tags": ["Inbox"],
            "summary": "Create routing rule",
            "operationId": "createInboxRoutingRules",
            "description": "Create a routing rule for inbox automation.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/RoutingRuleCreateRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response("Rule created", "RoutingRuleResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "get": {
            "tags": ["Inbox"],
            "summary": "List routing rules",
            "operationId": "listInboxRoutingRules",
            "description": "List routing rules for a workspace.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
                {"name": "enabled_only", "in": "query", "schema": {"type": "boolean"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer"}},
                {"name": "offset", "in": "query", "schema": {"type": "integer"}},
            ],
            "responses": {
                "200": _ok_response("Rule list", "RoutingRuleListResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/inbox/routing/rules/{id}": {
        "patch": {
            "tags": ["Inbox"],
            "summary": "Update routing rule",
            "operationId": "patchInboxRoutingRule",
            "description": "Update an inbox routing rule.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/RoutingRuleUpdateRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response("Rule updated", "RoutingRuleResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "delete": {
            "tags": ["Inbox"],
            "summary": "Delete routing rule",
            "operationId": "deleteInboxRoutingRule",
            "description": "Delete an inbox routing rule.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Rule deleted", "RoutingRuleDeleteResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/inbox/routing/rules/{id}/test": {
        "post": {
            "tags": ["Inbox"],
            "summary": "Test routing rule",
            "operationId": "createInboxRoutingRulesTest",
            "description": "Test a routing rule against existing messages.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/RoutingRuleTestRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response("Rule test", "RoutingRuleTestResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
}


__all__ = ["INBOX_ENDPOINTS"]
