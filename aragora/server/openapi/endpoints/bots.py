"""
OpenAPI endpoint definitions for Bot Platform Integrations.

Endpoints for chat platform webhooks and integrations including
Slack, Discord, Telegram, WhatsApp, Google Chat, Zoom, and Email.
"""

from aragora.server.openapi.helpers import (
    _ok_response,
    STANDARD_ERRORS,
)

BOTS_ENDPOINTS = {
    # Slack Bot Endpoints
    "/api/v1/bots/slack/events": {
        "post": {
            "tags": ["Bots - Slack"],
            "summary": "Handle Slack events",
            "description": """Webhook endpoint for Slack Events API.

Receives events from Slack including:
- Message events (new messages, edits, deletions)
- App mention events
- Reaction events
- Channel events (join, leave, rename)

**Verification:** Slack sends a challenge parameter for URL verification
which is echoed back.

**Signature:** All requests include X-Slack-Signature header for verification.""",
            "operationId": "handleSlackEvents",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "description": "Event type",
                                    "enum": ["url_verification", "event_callback"],
                                },
                                "challenge": {
                                    "type": "string",
                                    "description": "Challenge for URL verification",
                                },
                                "event": {
                                    "type": "object",
                                    "description": "Event payload",
                                },
                                "team_id": {"type": "string"},
                                "event_id": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response(
                    "Event processed",
                    {"ok": {"type": "boolean"}, "challenge": {"type": "string"}},
                ),
                "401": STANDARD_ERRORS["401"],
            },
        },
    },
    "/api/v1/bots/slack/interactions": {
        "post": {
            "tags": ["Bots - Slack"],
            "summary": "Handle Slack interactions",
            "description": """Webhook endpoint for Slack interactive components.

Handles user interactions including:
- Button clicks
- Menu selections
- Modal submissions
- Shortcut invocations
- Vote submissions for debates

**Payload:** Interactions are sent as form-encoded with a 'payload' JSON field.""",
            "operationId": "handleSlackInteractions",
            "requestBody": {
                "required": True,
                "content": {
                    "application/x-www-form-urlencoded": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "payload": {
                                    "type": "string",
                                    "description": "JSON-encoded interaction payload",
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Interaction handled", {}),
                "401": STANDARD_ERRORS["401"],
            },
        },
    },
    "/api/v1/bots/slack/commands": {
        "post": {
            "tags": ["Bots - Slack"],
            "summary": "Handle Slack slash commands",
            "description": """Webhook endpoint for Slack slash commands.

Handles the /aragora slash command for:
- Starting debates
- Checking debate status
- Voting on proposals
- Requesting summaries

**Response:** Returns immediate acknowledgment, with async updates via response_url.""",
            "operationId": "handleSlackCommands",
            "requestBody": {
                "required": True,
                "content": {
                    "application/x-www-form-urlencoded": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "command": {"type": "string", "example": "/aragora"},
                                "text": {"type": "string"},
                                "response_url": {"type": "string"},
                                "trigger_id": {"type": "string"},
                                "user_id": {"type": "string"},
                                "channel_id": {"type": "string"},
                                "team_id": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response(
                    "Command acknowledged",
                    {
                        "response_type": {
                            "type": "string",
                            "enum": ["ephemeral", "in_channel"],
                        },
                        "text": {"type": "string"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
            },
        },
    },
    "/api/v1/bots/slack/status": {
        "get": {
            "tags": ["Bots - Slack"],
            "summary": "Get Slack integration status",
            "description": """Returns the status of the Slack bot integration.

**Response includes:**
- Connection status
- Configured workspaces
- Active channels
- Recent activity metrics""",
            "operationId": "getSlackStatus",
            "responses": {
                "200": _ok_response(
                    "Slack integration status",
                    {
                        "connected": {"type": "boolean"},
                        "workspaces": {"type": "integer"},
                        "channels": {"type": "integer"},
                        "last_event": {"type": "string", "format": "date-time"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    # Discord Bot Endpoints
    "/api/v1/bots/discord/interactions": {
        "post": {
            "tags": ["Bots - Discord"],
            "summary": "Handle Discord interactions",
            "description": """Webhook endpoint for Discord Interactions API.

Handles all Discord interaction types:
- Slash commands
- Button clicks
- Select menu selections
- Modal submissions
- Autocomplete requests

**Verification:** Discord includes signature headers for request verification.""",
            "operationId": "handleDiscordInteractions",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "integer",
                                    "description": "Interaction type (1=Ping, 2=Command, 3=Component, 4=Autocomplete, 5=Modal)",
                                },
                                "data": {"type": "object"},
                                "guild_id": {"type": "string"},
                                "channel_id": {"type": "string"},
                                "member": {"type": "object"},
                                "token": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response(
                    "Interaction response",
                    {
                        "type": {"type": "integer"},
                        "data": {"type": "object"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
            },
        },
    },
    "/api/v1/bots/discord/status": {
        "get": {
            "tags": ["Bots - Discord"],
            "summary": "Get Discord integration status",
            "description": "Returns the status of the Discord bot integration.",
            "operationId": "getDiscordStatus",
            "responses": {
                "200": _ok_response(
                    "Discord integration status",
                    {
                        "connected": {"type": "boolean"},
                        "guilds": {"type": "integer"},
                        "users": {"type": "integer"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    # Telegram Bot Endpoints
    "/api/v1/bots/telegram/webhook": {
        "post": {
            "tags": ["Bots - Telegram"],
            "summary": "Handle Telegram webhook",
            "description": """Webhook endpoint for Telegram Bot API updates.

Receives updates including:
- New messages
- Edited messages
- Callback queries (inline button clicks)
- Inline queries

**Setup:** Configure webhook URL via Telegram Bot API setWebhook.""",
            "operationId": "handleTelegramWebhook",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "update_id": {"type": "integer"},
                                "message": {"type": "object"},
                                "callback_query": {"type": "object"},
                                "inline_query": {"type": "object"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Update processed", {}),
            },
        },
    },
    "/api/v1/bots/telegram/webhook/{token}": {
        "post": {
            "tags": ["Bots - Telegram"],
            "summary": "Handle Telegram webhook (token-verified)",
            "description": "Token-verified webhook endpoint for additional security.",
            "operationId": "handleTelegramWebhookToken",
            "parameters": [
                {
                    "name": "token",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Webhook verification token",
                }
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"type": "object"},
                    }
                },
            },
            "responses": {
                "200": _ok_response("Update processed", {}),
                "401": STANDARD_ERRORS["401"],
            },
        },
    },
    "/api/v1/bots/telegram/status": {
        "get": {
            "tags": ["Bots - Telegram"],
            "summary": "Get Telegram integration status",
            "operationId": "getTelegramStatus",
            "responses": {
                "200": _ok_response(
                    "Telegram integration status",
                    {
                        "connected": {"type": "boolean"},
                        "bot_username": {"type": "string"},
                        "chats": {"type": "integer"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    # WhatsApp Bot Endpoints
    "/api/v1/bots/whatsapp/webhook": {
        "post": {
            "tags": ["Bots - WhatsApp"],
            "summary": "Handle WhatsApp webhook",
            "description": """Webhook endpoint for WhatsApp Business API.

Receives notifications for:
- Incoming messages
- Message status updates
- Template message delivery

**Verification:** WhatsApp Cloud API includes hub.verify_token for setup.""",
            "operationId": "handleWhatsAppWebhook",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "object": {"type": "string"},
                                "entry": {
                                    "type": "array",
                                    "items": {"type": "object"},
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Webhook processed", {}),
            },
        },
        "get": {
            "tags": ["Bots - WhatsApp"],
            "summary": "Verify WhatsApp webhook",
            "description": "Webhook verification endpoint for WhatsApp setup.",
            "operationId": "verifyWhatsAppWebhook",
            "parameters": [
                {
                    "name": "hub.mode",
                    "in": "query",
                    "schema": {"type": "string"},
                },
                {
                    "name": "hub.verify_token",
                    "in": "query",
                    "schema": {"type": "string"},
                },
                {
                    "name": "hub.challenge",
                    "in": "query",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": {
                    "description": "Challenge response",
                    "content": {"text/plain": {"schema": {"type": "string"}}},
                },
            },
        },
    },
    "/api/v1/bots/whatsapp/status": {
        "get": {
            "tags": ["Bots - WhatsApp"],
            "summary": "Get WhatsApp integration status",
            "operationId": "getWhatsAppStatus",
            "responses": {
                "200": _ok_response(
                    "WhatsApp integration status",
                    {
                        "connected": {"type": "boolean"},
                        "phone_number": {"type": "string"},
                        "conversations": {"type": "integer"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    # Google Chat Bot Endpoints
    "/api/v1/bots/google-chat/webhook": {
        "post": {
            "tags": ["Bots - Google Chat"],
            "summary": "Handle Google Chat webhook",
            "description": """Webhook endpoint for Google Chat bot events.

Handles:
- MESSAGE events (new messages in spaces)
- ADDED_TO_SPACE events
- REMOVED_FROM_SPACE events
- CARD_CLICKED events (interactive card actions)""",
            "operationId": "handleGoogleChatWebhook",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "MESSAGE",
                                        "ADDED_TO_SPACE",
                                        "REMOVED_FROM_SPACE",
                                        "CARD_CLICKED",
                                    ],
                                },
                                "message": {"type": "object"},
                                "space": {"type": "object"},
                                "user": {"type": "object"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response(
                    "Response message",
                    {"text": {"type": "string"}, "cards": {"type": "array"}},
                ),
            },
        },
    },
    "/api/v1/bots/google-chat/status": {
        "get": {
            "tags": ["Bots - Google Chat"],
            "summary": "Get Google Chat integration status",
            "operationId": "getGoogleChatStatus",
            "responses": {
                "200": _ok_response(
                    "Google Chat integration status",
                    {
                        "connected": {"type": "boolean"},
                        "spaces": {"type": "integer"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    # Zoom Bot Endpoints
    "/api/v1/bots/zoom/events": {
        "post": {
            "tags": ["Bots - Zoom"],
            "summary": "Handle Zoom events",
            "description": """Webhook endpoint for Zoom app events.

Handles events including:
- meeting.started / meeting.ended
- chat_message.sent
- bot_notification

**Verification:** Zoom includes verification token in headers.""",
            "operationId": "handleZoomEvents",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "event": {"type": "string"},
                                "payload": {"type": "object"},
                                "event_ts": {"type": "integer"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Event processed", {}),
            },
        },
    },
    "/api/v1/bots/zoom/status": {
        "get": {
            "tags": ["Bots - Zoom"],
            "summary": "Get Zoom integration status",
            "operationId": "getZoomStatus",
            "responses": {
                "200": _ok_response(
                    "Zoom integration status",
                    {
                        "connected": {"type": "boolean"},
                        "account_id": {"type": "string"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    # Email Webhook Endpoints
    "/api/v1/bots/email/webhook/sendgrid": {
        "post": {
            "tags": ["Bots - Email"],
            "summary": "Handle SendGrid inbound email",
            "description": """Webhook endpoint for SendGrid Inbound Parse.

Receives parsed email data including:
- From, To, Subject headers
- Plain text and HTML body
- Attachments (as multipart form data)""",
            "operationId": "handleSendGridWebhook",
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "from": {"type": "string"},
                                "to": {"type": "string"},
                                "subject": {"type": "string"},
                                "text": {"type": "string"},
                                "html": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Email processed", {}),
            },
        },
    },
    "/api/v1/bots/email/webhook/ses": {
        "post": {
            "tags": ["Bots - Email"],
            "summary": "Handle AWS SES notifications",
            "description": """Webhook endpoint for AWS SES notifications via SNS.

Handles notification types:
- Delivery notifications
- Bounce notifications
- Complaint notifications""",
            "operationId": "handleSESWebhook",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "Type": {"type": "string"},
                                "Message": {"type": "string"},
                                "MessageId": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Notification processed", {}),
            },
        },
    },
    "/api/v1/bots/email/status": {
        "get": {
            "tags": ["Bots - Email"],
            "summary": "Get email integration status",
            "operationId": "getEmailStatus",
            "responses": {
                "200": _ok_response(
                    "Email integration status",
                    {
                        "sendgrid_connected": {"type": "boolean"},
                        "ses_connected": {"type": "boolean"},
                        "emails_processed": {"type": "integer"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
}
