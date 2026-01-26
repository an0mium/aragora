"""Gmail operations endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS, AUTH_REQUIREMENTS

GMAIL_ENDPOINTS = {
    "/api/v1/gmail/labels": {
        "get": {
            "tags": ["Email"],
            "summary": "List Gmail labels",
            "operationId": "listGmailLabels",
            "description": "List Gmail labels for a connected account.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "user_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Labels", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "post": {
            "tags": ["Email"],
            "summary": "Create Gmail label",
            "operationId": "createGmailLabels",
            "description": "Create a Gmail label.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "user_id": {"type": "string"},
                                "message_list_visibility": {"type": "string"},
                                "label_list_visibility": {"type": "string"},
                            },
                            "required": ["name"],
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Label created", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/gmail/labels/{label_id}": {
        "patch": {
            "tags": ["Email"],
            "summary": "Update Gmail label",
            "operationId": "patchGmailLabel",
            "description": "Update a Gmail label.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "label_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "user_id": {"type": "string"},
                                "message_list_visibility": {"type": "string"},
                                "label_list_visibility": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Label updated", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "delete": {
            "tags": ["Email"],
            "summary": "Delete Gmail label",
            "operationId": "deleteGmailLabel",
            "description": "Delete a Gmail label.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "label_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {"name": "user_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Label deleted", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/gmail/filters": {
        "get": {
            "tags": ["Email"],
            "summary": "List Gmail filters",
            "operationId": "listGmailFilters",
            "description": "List Gmail filters.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "user_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Filters", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "post": {
            "tags": ["Email"],
            "summary": "Create Gmail filter",
            "operationId": "createGmailFilters",
            "description": "Create a Gmail filter.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Filter created", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/gmail/filters/{filter_id}": {
        "delete": {
            "tags": ["Email"],
            "summary": "Delete Gmail filter",
            "operationId": "deleteGmailFilter",
            "description": "Delete a Gmail filter.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "filter_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {"name": "user_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Filter deleted", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/gmail/messages/{message_id}/labels": {
        "post": {
            "tags": ["Email"],
            "summary": "Modify message labels",
            "operationId": "createGmailMessagesLabel",
            "description": "Add or remove labels for a message.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "message_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Labels updated", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/gmail/messages/{message_id}/read": {
        "post": {
            "tags": ["Email"],
            "summary": "Mark message read/unread",
            "operationId": "createGmailMessagesRead",
            "description": "Toggle read state for a message.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "message_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
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
    "/api/v1/gmail/messages/{message_id}/star": {
        "post": {
            "tags": ["Email"],
            "summary": "Star or unstar message",
            "operationId": "createGmailMessagesStar",
            "description": "Toggle star state for a message.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "message_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
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
    "/api/v1/gmail/messages/{message_id}/archive": {
        "post": {
            "tags": ["Email"],
            "summary": "Archive message",
            "operationId": "createGmailMessagesArchive",
            "description": "Archive a message.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "message_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Message archived", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/gmail/messages/{message_id}/trash": {
        "post": {
            "tags": ["Email"],
            "summary": "Trash message",
            "operationId": "createGmailMessagesTrash",
            "description": "Trash or untrash a message.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "message_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "requestBody": {
                "required": False,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Message trashed", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/gmail/messages/{message_id}/attachments/{attachment_id}": {
        "get": {
            "tags": ["Email"],
            "summary": "Get attachment",
            "operationId": "getGmailMessagesAttachment",
            "description": "Fetch a message attachment.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "message_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "attachment_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {"name": "user_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Attachment", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/gmail/threads": {
        "get": {
            "tags": ["Email"],
            "summary": "List Gmail threads",
            "operationId": "listGmailThreads",
            "description": "List Gmail threads.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "user_id", "in": "query", "schema": {"type": "string"}},
                {"name": "q", "in": "query", "schema": {"type": "string"}},
                {"name": "label_ids", "in": "query", "schema": {"type": "string"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer"}},
                {"name": "page_token", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Threads", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/gmail/threads/{thread_id}": {
        "get": {
            "tags": ["Email"],
            "summary": "Get Gmail thread",
            "operationId": "getGmailThread",
            "description": "Fetch a thread with messages.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "thread_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {"name": "user_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Thread", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/gmail/threads/{thread_id}/archive": {
        "post": {
            "tags": ["Email"],
            "summary": "Archive thread",
            "operationId": "createGmailThreadsArchive",
            "description": "Archive a Gmail thread.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "thread_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Thread archived", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/gmail/threads/{thread_id}/trash": {
        "post": {
            "tags": ["Email"],
            "summary": "Trash thread",
            "operationId": "createGmailThreadsTrash",
            "description": "Trash or untrash a thread.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "thread_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "requestBody": {
                "required": False,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Thread trashed", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/gmail/threads/{thread_id}/labels": {
        "post": {
            "tags": ["Email"],
            "summary": "Modify thread labels",
            "operationId": "createGmailThreadsLabel",
            "description": "Add or remove labels on a thread.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "thread_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Thread updated", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/gmail/drafts": {
        "get": {
            "tags": ["Email"],
            "summary": "List drafts",
            "operationId": "listGmailDrafts",
            "description": "List Gmail drafts.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "user_id", "in": "query", "schema": {"type": "string"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer"}},
                {"name": "page_token", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Drafts", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "post": {
            "tags": ["Email"],
            "summary": "Create draft",
            "operationId": "createGmailDrafts",
            "description": "Create a Gmail draft.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Draft created", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/gmail/drafts/{draft_id}": {
        "get": {
            "tags": ["Email"],
            "summary": "Get draft",
            "operationId": "getGmailDraft",
            "description": "Get a Gmail draft by ID.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "draft_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {"name": "user_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Draft", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "put": {
            "tags": ["Email"],
            "summary": "Update draft",
            "operationId": "updateGmailDraft",
            "description": "Update a Gmail draft.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "draft_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Draft updated", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "delete": {
            "tags": ["Email"],
            "summary": "Delete draft",
            "operationId": "deleteGmailDraft",
            "description": "Delete a Gmail draft.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "draft_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {"name": "user_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Draft deleted", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/gmail/drafts/{draft_id}/send": {
        "post": {
            "tags": ["Email"],
            "summary": "Send draft",
            "operationId": "createGmailDraftsSend",
            "description": "Send a Gmail draft.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "draft_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Draft sent", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
}


__all__ = ["GMAIL_ENDPOINTS"]
