"""SDK missing endpoints for Integrations, Webhooks, and Connectors.

This module contains endpoint definitions for external integrations,
webhook management, and connector configuration.
"""

from aragora.server.openapi.endpoints.sdk_missing_core import (
    _ok_response,
    STANDARD_ERRORS,
)

SDK_MISSING_INTEGRATION_ENDPOINTS: dict = {
    # Integrations endpoints
    "/api/integrations": {
        "get": {
            "tags": ["Integrations"],
            "summary": "GET integrations",
            "operationId": "getIntegrations",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Integrations"],
            "summary": "POST integrations",
            "operationId": "postIntegrations",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/integrations/available": {
        "get": {
            "tags": ["Integrations"],
            "summary": "GET available",
            "operationId": "getIntegrationsAvailable",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/integrations/config/{id}": {
        "get": {
            "tags": ["Integrations"],
            "summary": "GET {id}",
            "operationId": "getIntegrationsConfig",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/integrations/discord/callback": {
        "post": {
            "tags": ["Integrations"],
            "summary": "POST callback",
            "operationId": "postDiscordCallback",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/integrations/discord/install": {
        "post": {
            "tags": ["Integrations"],
            "summary": "POST install",
            "operationId": "postDiscordInstall",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/integrations/teams/callback": {
        "post": {
            "tags": ["Integrations"],
            "summary": "POST callback",
            "operationId": "postTeamsCallback",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/integrations/teams/install": {
        "post": {
            "tags": ["Integrations"],
            "summary": "POST install",
            "operationId": "postTeamsInstall",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/integrations/{id}": {
        "delete": {
            "tags": ["Integrations"],
            "summary": "DELETE {id}",
            "operationId": "deleteIntegrations",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "get": {
            "tags": ["Integrations"],
            "summary": "GET {id}",
            "operationId": "getIntegrations",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "put": {
            "tags": ["Integrations"],
            "summary": "PUT {id}",
            "operationId": "putIntegrations",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/integrations/{id}/sync": {
        "get": {
            "tags": ["Integrations"],
            "summary": "GET sync",
            "operationId": "getIntegrationsSync",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Integrations"],
            "summary": "POST sync",
            "operationId": "postIntegrationsSync",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/integrations/{id}/test": {
        "post": {
            "tags": ["Integrations"],
            "summary": "POST test",
            "operationId": "postIntegrationsTest",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    # Connectors endpoints
    "/api/v1/connectors": {
        "post": {
            "tags": ["Connectors"],
            "summary": "POST connectors",
            "operationId": "postConnectors",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/connectors/{id}": {
        "delete": {
            "tags": ["Connectors"],
            "summary": "DELETE {id}",
            "operationId": "deleteConnectors",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "patch": {
            "tags": ["Connectors"],
            "summary": "PATCH {id}",
            "operationId": "patchConnectors",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/connectors/{id}/health": {
        "get": {
            "tags": ["Connectors"],
            "summary": "GET health",
            "operationId": "getConnectorsHealth",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/connectors/{id}/sync": {
        "post": {
            "tags": ["Connectors"],
            "summary": "POST sync",
            "operationId": "postConnectorsSync",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/connectors/{id}/syncs": {
        "get": {
            "tags": ["Connectors"],
            "summary": "GET syncs",
            "operationId": "getConnectorsSyncs",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/connectors/{id}/syncs/{id}": {
        "get": {
            "tags": ["Connectors"],
            "summary": "GET sync by ID",
            "operationId": "getConnectorSyncById",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "id2", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/connectors/{id}/syncs/{id}/cancel": {
        "post": {
            "tags": ["Connectors"],
            "summary": "POST cancel",
            "operationId": "postSyncsCancel",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "id2", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/connectors/{id}/test": {
        "post": {
            "tags": ["Connectors"],
            "summary": "POST test",
            "operationId": "postConnectorsTest",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    # Webhooks endpoints
    "/api/v1/webhooks/bulk": {
        "delete": {
            "tags": ["Webhooks"],
            "summary": "DELETE bulk",
            "operationId": "deleteWebhooksBulk",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/webhooks/events/categories": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "GET categories",
            "operationId": "getEventsCategories",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/webhooks/pause-all": {
        "post": {
            "tags": ["Webhooks"],
            "summary": "POST pause-all",
            "operationId": "postWebhooksPause-All",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/webhooks/resume-all": {
        "post": {
            "tags": ["Webhooks"],
            "summary": "POST resume-all",
            "operationId": "postWebhooksResume-All",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/webhooks/{id}/deliveries": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "GET deliveries",
            "operationId": "getWebhooksDeliveries",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/webhooks/{id}/deliveries/{id}": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "GET delivery by ID",
            "operationId": "getWebhookDeliveryById",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "id2", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/webhooks/{id}/deliveries/{id}/retry": {
        "post": {
            "tags": ["Webhooks"],
            "summary": "POST retry",
            "operationId": "postDeliveriesRetry",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "id2", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/webhooks/{id}/events": {
        "delete": {
            "tags": ["Webhooks"],
            "summary": "DELETE events",
            "operationId": "deleteWebhooksEvents",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "post": {
            "tags": ["Webhooks"],
            "summary": "POST events",
            "operationId": "postWebhooksEvents",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/webhooks/{id}/retry-policy": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "GET retry-policy",
            "operationId": "getWebhooksRetry-Policy",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "put": {
            "tags": ["Webhooks"],
            "summary": "PUT retry-policy",
            "operationId": "putWebhooksRetry-Policy",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/webhooks/{id}/rotate-secret": {
        "post": {
            "tags": ["Webhooks"],
            "summary": "POST rotate-secret",
            "operationId": "postWebhooksRotate-Secret",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/webhooks/{id}/signing": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "GET signing",
            "operationId": "getWebhooksSigning",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/webhooks/{id}/stats": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "GET stats",
            "operationId": "getWebhooksStats",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
}

__all__ = ["SDK_MISSING_INTEGRATION_ENDPOINTS"]
