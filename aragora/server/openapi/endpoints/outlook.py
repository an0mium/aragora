"""Outlook/M365 endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS, AUTH_REQUIREMENTS

OUTLOOK_ENDPOINTS = {
    "/api/v1/outlook/oauth/url": {
        "get": {
            "tags": ["Email"],
            "summary": "Get OAuth URL",
            "operationId": "listOutlookOauthUrl",
            "description": "Generate Outlook OAuth authorization URL.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "redirect_uri",
                    "in": "query",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("OAuth URL", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/outlook/oauth/callback": {
        "post": {
            "tags": ["Email"],
            "summary": "OAuth callback",
            "operationId": "createOutlookOauthCallback",
            "description": "Handle Outlook OAuth callback.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string"},
                                "state": {"type": "string"},
                                "redirect_uri": {"type": "string"},
                            },
                            "required": ["code", "state"],
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("OAuth complete", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/outlook/folders": {
        "get": {
            "tags": ["Email"],
            "summary": "List folders",
            "operationId": "listOutlookFolders",
            "description": "List Outlook mail folders.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Folders", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/outlook/messages": {
        "get": {
            "tags": ["Email"],
            "summary": "List messages",
            "operationId": "listOutlookMessages",
            "description": "List Outlook messages.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
                {"name": "folder_id", "in": "query", "schema": {"type": "string"}},
                {"name": "max_results", "in": "query", "schema": {"type": "integer"}},
                {"name": "page_token", "in": "query", "schema": {"type": "string"}},
                {"name": "filter", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Messages", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/outlook/messages/{message_id}": {
        "get": {
            "tags": ["Email"],
            "summary": "Get message",
            "operationId": "getOutlookMessage",
            "description": "Fetch Outlook message details.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "message_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Message", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "delete": {
            "tags": ["Email"],
            "summary": "Delete message",
            "operationId": "deleteOutlookMessage",
            "description": "Delete an Outlook message.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "message_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Message deleted", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/outlook/conversations/{conversation_id}": {
        "get": {
            "tags": ["Email"],
            "summary": "Get conversation",
            "operationId": "getOutlookConversation",
            "description": "Fetch Outlook conversation thread.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "conversation_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Conversation", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/outlook/send": {
        "post": {
            "tags": ["Email"],
            "summary": "Send message",
            "operationId": "createOutlookSend",
            "description": "Send a new Outlook message.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Message sent", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/outlook/reply": {
        "post": {
            "tags": ["Email"],
            "summary": "Reply to message",
            "operationId": "createOutlookReply",
            "description": "Reply to an Outlook message.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Reply sent", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/outlook/search": {
        "get": {
            "tags": ["Email"],
            "summary": "Search messages",
            "operationId": "listOutlookSearch",
            "description": "Search Outlook messages with OData filters.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "query", "in": "query", "schema": {"type": "string"}},
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Search results", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/outlook/messages/{message_id}/read": {
        "post": {
            "tags": ["Email"],
            "summary": "Mark read/unread",
            "operationId": "createOutlookMessagesRead",
            "description": "Toggle read state for an Outlook message.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "message_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "requestBody": {
                "required": False,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Message updated", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/outlook/messages/{message_id}/move": {
        "post": {
            "tags": ["Email"],
            "summary": "Move message",
            "operationId": "createOutlookMessagesMove",
            "description": "Move a message to a folder.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "message_id",
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
                "200": _ok_response("Message moved", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/outlook/status": {
        "get": {
            "tags": ["Email"],
            "summary": "Connection status",
            "operationId": "listOutlookStatus",
            "description": "Get Outlook integration status.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Status", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
}


__all__ = ["OUTLOOK_ENDPOINTS"]
