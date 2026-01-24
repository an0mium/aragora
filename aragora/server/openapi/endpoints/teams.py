"""
Microsoft Teams Integration API Endpoints.

OpenAPI 3.1 specification for Teams integration endpoints.
"""

from aragora.server.openapi.schemas import STANDARD_ERRORS

TEAMS_ENDPOINTS = {
    "/api/v1/integrations/teams/status": {
        "get": {
            "tags": ["Teams", "Integrations"],
            "summary": "Get Teams integration status",
            "description": "Check if Microsoft Teams integration is configured and ready.",
            "operationId": "getTeamsStatus",
            "responses": {
                "200": {
                    "description": "Teams integration status",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "enabled": {
                                        "type": "boolean",
                                        "description": "Whether Teams integration is enabled",
                                    },
                                    "app_id_configured": {
                                        "type": "boolean",
                                        "description": "Whether TEAMS_APP_ID is set",
                                    },
                                    "password_configured": {
                                        "type": "boolean",
                                        "description": "Whether TEAMS_APP_PASSWORD is set",
                                    },
                                    "tenant_id_configured": {
                                        "type": "boolean",
                                        "description": "Whether TEAMS_TENANT_ID is set",
                                    },
                                    "connector_ready": {
                                        "type": "boolean",
                                        "description": "Whether the connector is initialized",
                                    },
                                },
                            },
                            "example": {
                                "enabled": True,
                                "app_id_configured": True,
                                "password_configured": True,
                                "tenant_id_configured": True,
                                "connector_ready": True,
                            },
                        }
                    },
                },
            },
        },
    },
    "/api/v1/integrations/teams/commands": {
        "post": {
            "tags": ["Teams", "Integrations"],
            "summary": "Handle Teams bot command",
            "description": """
Handle @aragora commands from Microsoft Teams. Commands are:
- `@aragora debate <topic>` - Start a new debate
- `@aragora status` - Check active debate status
- `@aragora cancel` - Cancel the active debate
- `@aragora help` - Show available commands
""",
            "operationId": "handleTeamsCommand",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "description": "Bot Framework Activity object",
                            "properties": {
                                "type": {"type": "string"},
                                "text": {"type": "string"},
                                "conversation": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "name": {"type": "string"},
                                    },
                                },
                                "serviceUrl": {"type": "string"},
                                "from": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "name": {"type": "string"},
                                    },
                                },
                            },
                        },
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Command processed",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "success": {"type": "boolean"},
                                    "message": {"type": "string"},
                                    "conversation_id": {"type": "string"},
                                    "topic": {"type": "string"},
                                },
                            },
                        }
                    },
                },
                **STANDARD_ERRORS,
            },
        },
    },
    "/api/v1/integrations/teams/interactive": {
        "post": {
            "tags": ["Teams", "Integrations"],
            "summary": "Handle Teams Adaptive Card action",
            "description": "Handle interactive actions from Adaptive Cards (votes, button clicks).",
            "operationId": "handleTeamsInteractive",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["invoke"]},
                                "value": {
                                    "type": "object",
                                    "properties": {
                                        "action": {"type": "string"},
                                        "vote": {"type": "string"},
                                        "debate_id": {"type": "string"},
                                        "receipt_id": {"type": "string"},
                                    },
                                },
                                "conversation": {"type": "object"},
                                "serviceUrl": {"type": "string"},
                            },
                        },
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Action processed",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "status": {"type": "string"},
                                    "vote": {"type": "string"},
                                },
                            },
                        }
                    },
                },
                **STANDARD_ERRORS,
            },
        },
    },
    "/api/v1/integrations/teams/notify": {
        "post": {
            "tags": ["Teams", "Integrations"],
            "summary": "Send notification to Teams channel",
            "description": "Send a message or Adaptive Card to a Teams conversation.",
            "operationId": "sendTeamsNotification",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "conversation_id": {
                                    "type": "string",
                                    "description": "Teams conversation ID",
                                },
                                "service_url": {
                                    "type": "string",
                                    "description": "Bot Framework service URL",
                                },
                                "message": {
                                    "type": "string",
                                    "description": "Message text",
                                },
                                "blocks": {
                                    "type": "array",
                                    "description": "Adaptive Card body elements",
                                    "items": {"type": "object"},
                                },
                            },
                            "required": ["conversation_id", "service_url"],
                        },
                        "example": {
                            "conversation_id": "19:abc123@thread.v2",
                            "service_url": "https://smba.trafficmanager.net/emea/",
                            "message": "Debate complete!",
                            "blocks": [
                                {
                                    "type": "TextBlock",
                                    "text": "Decision Made",
                                    "size": "Large",
                                }
                            ],
                        },
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Notification sent",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "success": {"type": "boolean"},
                                    "message_id": {"type": "string"},
                                    "error": {"type": ["string", "null"]},
                                },
                            },
                        }
                    },
                },
                **STANDARD_ERRORS,
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/bots/teams/messages": {
        "post": {
            "tags": ["Teams", "Bots"],
            "summary": "Bot Framework messaging endpoint",
            "description": "Receive incoming activities from Microsoft Teams via Bot Framework.",
            "operationId": "handleTeamsMessages",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "description": "Bot Framework Activity",
                        },
                    }
                },
            },
            "responses": {
                "200": {"description": "Activity processed"},
                "401": {"description": "Invalid authentication"},
                **STANDARD_ERRORS,
            },
        },
    },
    "/api/v1/bots/teams/status": {
        "get": {
            "tags": ["Teams", "Bots"],
            "summary": "Bot Framework status",
            "description": "Check if Bot Framework SDK and credentials are configured.",
            "operationId": "getTeamsBotStatus",
            "responses": {
                "200": {
                    "description": "Bot status",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "enabled": {"type": "boolean"},
                                    "app_id_configured": {"type": "boolean"},
                                    "password_configured": {"type": "boolean"},
                                    "sdk_available": {"type": "boolean"},
                                    "sdk_error": {"type": ["string", "null"]},
                                },
                            },
                        }
                    },
                },
            },
        },
    },
}

__all__ = ["TEAMS_ENDPOINTS"]
