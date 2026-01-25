"""Webhook endpoint definitions for OpenAPI documentation."""

from aragora.server.openapi.helpers import _ok_response

# Common parameter definitions
_WEBHOOK_ID_PARAM = {
    "name": "id",
    "in": "path",
    "required": True,
    "schema": {"type": "string", "format": "uuid"},
    "description": "Webhook ID",
}


def _webhook_schema() -> dict:
    """Webhook object schema."""
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string", "format": "uuid", "description": "Unique webhook ID"},
            "name": {"type": "string", "description": "Human-readable webhook name"},
            "url": {"type": "string", "format": "uri", "description": "Webhook delivery URL"},
            "events": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of event types to subscribe to",
            },
            "enabled": {"type": "boolean", "description": "Whether webhook is active"},
            "created_at": {"type": "string", "format": "date-time"},
            "updated_at": {"type": "string", "format": "date-time"},
            "secret": {
                "type": "string",
                "description": "HMAC secret for signature verification (only shown on creation)",
            },
        },
        "required": ["id", "name", "url", "events", "enabled"],
    }


def _webhook_create_schema() -> dict:
    """Webhook creation request schema."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Human-readable webhook name"},
            "url": {"type": "string", "format": "uri", "description": "Webhook delivery URL"},
            "events": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of event types to subscribe to",
            },
            "enabled": {"type": "boolean", "default": True},
        },
        "required": ["name", "url", "events"],
    }


def _webhook_update_schema() -> dict:
    """Webhook update request schema."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "url": {"type": "string", "format": "uri"},
            "events": {"type": "array", "items": {"type": "string"}},
            "enabled": {"type": "boolean"},
        },
    }


def _event_type_schema() -> dict:
    """Event type schema."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Event type name"},
            "description": {"type": "string", "description": "Event description"},
            "category": {"type": "string", "description": "Event category"},
        },
    }


WEBHOOK_ENDPOINTS = {
    "/api/v1/webhooks": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "List webhooks",
            "description": "List all webhooks for the authenticated user/organization.",
            "responses": {
                "200": {
                    "description": "List of webhooks",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "webhooks": {
                                        "type": "array",
                                        "items": _webhook_schema(),
                                    },
                                    "total": {"type": "integer"},
                                },
                            }
                        }
                    },
                }
            },
            "security": [{"bearerAuth": []}],
        },
        "post": {
            "tags": ["Webhooks"],
            "summary": "Create webhook",
            "description": "Register a new webhook to receive event notifications.",
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": _webhook_create_schema()}},
            },
            "responses": {
                "201": {
                    "description": "Webhook created successfully",
                    "content": {
                        "application/json": {
                            "schema": _webhook_schema(),
                        }
                    },
                },
                "400": _ok_response("Invalid webhook configuration"),
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/webhooks/events": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "List available event types",
            "description": "Get all event types that can be subscribed to via webhooks.",
            "responses": {
                "200": {
                    "description": "List of event types",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "events": {
                                        "type": "array",
                                        "items": _event_type_schema(),
                                    }
                                },
                            }
                        }
                    },
                }
            },
        },
    },
    "/api/v1/webhooks/{id}": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "Get webhook",
            "description": "Get details of a specific webhook.",
            "parameters": [_WEBHOOK_ID_PARAM],
            "responses": {
                "200": {
                    "description": "Webhook details",
                    "content": {"application/json": {"schema": _webhook_schema()}},
                },
                "404": _ok_response("Webhook not found"),
            },
            "security": [{"bearerAuth": []}],
        },
        "patch": {
            "tags": ["Webhooks"],
            "summary": "Update webhook",
            "description": "Update webhook configuration.",
            "parameters": [_WEBHOOK_ID_PARAM],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": _webhook_update_schema()}},
            },
            "responses": {
                "200": {
                    "description": "Webhook updated",
                    "content": {"application/json": {"schema": _webhook_schema()}},
                },
                "404": _ok_response("Webhook not found"),
            },
            "security": [{"bearerAuth": []}],
        },
        "delete": {
            "tags": ["Webhooks"],
            "summary": "Delete webhook",
            "description": "Delete a webhook subscription.",
            "parameters": [_WEBHOOK_ID_PARAM],
            "responses": {
                "204": {"description": "Webhook deleted"},
                "404": _ok_response("Webhook not found"),
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/webhooks/{id}/test": {
        "post": {
            "tags": ["Webhooks"],
            "summary": "Test webhook",
            "description": "Send a test event to verify webhook configuration.",
            "parameters": [_WEBHOOK_ID_PARAM],
            "requestBody": {
                "required": False,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "event_type": {
                                    "type": "string",
                                    "description": "Event type to simulate (default: test)",
                                }
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Test delivery result",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "success": {"type": "boolean"},
                                    "status_code": {"type": "integer"},
                                    "response_time_ms": {"type": "number"},
                                    "error": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "404": _ok_response("Webhook not found"),
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/webhooks/slo/status": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "Get SLO webhook status",
            "description": "Get status of SLO (Service Level Objective) webhook notifications.",
            "responses": {
                "200": {
                    "description": "SLO webhook status",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "enabled": {"type": "boolean"},
                                    "url": {"type": "string", "format": "uri"},
                                    "last_delivery": {"type": "string", "format": "date-time"},
                                    "delivery_success_rate": {"type": "number"},
                                },
                            }
                        }
                    },
                }
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/webhooks/slo/test": {
        "post": {
            "tags": ["Webhooks"],
            "summary": "Test SLO webhook",
            "description": "Send a test SLO violation notification.",
            "responses": {
                "200": {
                    "description": "Test delivery result",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "success": {"type": "boolean"},
                                    "message": {"type": "string"},
                                },
                            }
                        }
                    },
                }
            },
            "security": [{"bearerAuth": []}],
        },
    },
}
