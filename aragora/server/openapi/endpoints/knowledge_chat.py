"""Knowledge chat bridge endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS, AUTH_REQUIREMENTS


KNOWLEDGE_CHAT_ENDPOINTS = {
    "/api/v1/chat/knowledge/search": {
        "post": {
            "tags": ["Knowledge Mound"],
            "summary": "Search knowledge",
            "operationId": "createChatKnowledgeSearch",
            "description": "Search knowledge from chat context.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "workspace_id": {"type": "string"},
                                "channel_id": {"type": "string"},
                                "scope": {"type": "string"},
                                "strategy": {"type": "string"},
                                "node_types": {"type": "array", "items": {"type": "string"}},
                                "max_results": {"type": "integer"},
                            },
                            "required": ["query"],
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Search results", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/chat/knowledge/inject": {
        "post": {
            "tags": ["Knowledge Mound"],
            "summary": "Inject knowledge",
            "operationId": "createChatKnowledgeInject",
            "description": "Retrieve knowledge context for a conversation.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Injected context", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/chat/knowledge/store": {
        "post": {
            "tags": ["Knowledge Mound"],
            "summary": "Store chat knowledge",
            "operationId": "createChatKnowledgeStore",
            "description": "Persist chat conversation as knowledge.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Knowledge stored", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/chat/knowledge/channel/{channel_id}/summary": {
        "get": {
            "tags": ["Knowledge Mound"],
            "summary": "Channel summary",
            "operationId": "getChatKnowledgeChannelSummary",
            "description": "Get channel knowledge summary.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "channel_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Channel summary", "StandardSuccessResponse"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
}


__all__ = ["KNOWLEDGE_CHAT_ENDPOINTS"]
