"""Webhook endpoint definitions for OpenAPI documentation."""

from typing import Any

from aragora.server.openapi.helpers import _ok_response

# Common parameter definitions
_WEBHOOK_ID_PARAM = {
    "name": "id",
    "in": "path",
    "required": True,
    "schema": {"type": "string", "format": "uuid"},
    "description": "Webhook ID",
}


def _webhook_schema() -> dict[str, Any]:
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


def _webhook_create_schema() -> dict[str, Any]:
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


def _webhook_update_schema() -> dict[str, Any]:
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


def _event_type_schema() -> dict[str, Any]:
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
            "operationId": "listWebhooks",
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
            "operationId": "createWebhooks",
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
            "operationId": "listWebhooksEvents",
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
            "operationId": "getWebhook",
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
            "operationId": "patchWebhook",
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
            "operationId": "deleteWebhook",
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
            "operationId": "createWebhooksTest",
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
            "operationId": "listWebhooksSloStatus",
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
            "operationId": "createWebhooksSloTest",
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
    "/api/v1/webhooks/dead-letter": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "List dead-letter deliveries",
            "operationId": "listWebhooksDeadLetter",
            "description": "List webhook deliveries that failed and are in the dead-letter queue.",
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 100},
                    "description": "Maximum number of deliveries to return",
                }
            ],
            "responses": {
                "200": {
                    "description": "List of dead-letter deliveries",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "dead_letters": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "webhook_id": {"type": "string"},
                                                "event_type": {"type": "string"},
                                                "payload": {"type": "object"},
                                                "error": {"type": "string"},
                                                "attempts": {"type": "integer"},
                                                "created_at": {
                                                    "type": "string",
                                                    "format": "date-time",
                                                },
                                                "last_attempt": {
                                                    "type": "string",
                                                    "format": "date-time",
                                                },
                                            },
                                        },
                                    },
                                    "count": {"type": "integer"},
                                },
                            }
                        }
                    },
                }
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/webhooks/dead-letter/{id}": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "Get dead-letter delivery",
            "operationId": "getWebhooksDeadLetter",
            "description": "Get details of a specific dead-letter delivery.",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": {
                    "description": "Dead-letter delivery details",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "webhook_id": {"type": "string"},
                                    "event_type": {"type": "string"},
                                    "payload": {"type": "object"},
                                    "error": {"type": "string"},
                                    "attempts": {"type": "integer"},
                                    "created_at": {"type": "string", "format": "date-time"},
                                    "last_attempt": {"type": "string", "format": "date-time"},
                                },
                            }
                        }
                    },
                },
                "404": _ok_response("Delivery not found"),
            },
            "security": [{"bearerAuth": []}],
        },
        "delete": {
            "tags": ["Webhooks"],
            "summary": "Delete dead-letter delivery",
            "operationId": "deleteWebhooksDeadLetter",
            "description": "Remove a delivery from the dead-letter queue without retrying.",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": {"description": "Delivery deleted"},
                "404": _ok_response("Delivery not found"),
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/webhooks/dead-letter/{id}/retry": {
        "post": {
            "tags": ["Webhooks"],
            "summary": "Retry dead-letter delivery",
            "operationId": "retryWebhooksDeadLetter",
            "description": "Retry a dead-letter delivery by moving it back to the processing queue.",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": {"description": "Delivery queued for retry"},
                "404": _ok_response("Delivery not found or not in dead-letter queue"),
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/webhooks/queue/stats": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "Get webhook queue statistics",
            "operationId": "getWebhooksQueueStats",
            "description": "Get statistics about the webhook delivery queue including pending, processing, and dead-letter counts.",
            "responses": {
                "200": {
                    "description": "Queue statistics",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "stats": {
                                        "type": "object",
                                        "properties": {
                                            "pending": {"type": "integer"},
                                            "processing": {"type": "integer"},
                                            "dead_letter": {"type": "integer"},
                                            "total_delivered": {"type": "integer"},
                                            "total_failed": {"type": "integer"},
                                        },
                                    }
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
