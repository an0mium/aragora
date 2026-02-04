"""SDK missing endpoints for Integrations, Webhooks, and Connectors.

This module contains endpoint definitions for external integrations,
webhook management, and connector configuration.
"""

from aragora.server.openapi.endpoints.sdk_missing_core import (
    STANDARD_ERRORS,
    _ok_response,
)

# =============================================================================
# Response Schemas
# =============================================================================

_INTEGRATION_SCHEMA = {
    "id": {"type": "string", "description": "Unique integration identifier"},
    "name": {"type": "string", "description": "Integration display name"},
    "type": {
        "type": "string",
        "enum": ["slack", "teams", "discord", "email", "zapier", "n8n", "custom"],
        "description": "Integration type",
    },
    "status": {
        "type": "string",
        "enum": ["active", "inactive", "error", "pending"],
        "description": "Current integration status",
    },
    "config": {
        "type": "object",
        "description": "Integration-specific configuration",
    },
    "workspace_id": {"type": "string", "description": "Associated workspace"},
    "created_at": {"type": "string", "format": "date-time"},
    "updated_at": {"type": "string", "format": "date-time"},
    "last_sync": {"type": "string", "format": "date-time", "description": "Last successful sync"},
}

_INTEGRATION_LIST_SCHEMA = {
    "integrations": {
        "type": "array",
        "items": {"type": "object", "properties": _INTEGRATION_SCHEMA},
        "description": "List of integrations",
    },
    "total": {"type": "integer", "description": "Total number of integrations"},
}

_AVAILABLE_INTEGRATION_SCHEMA = {
    "type": {"type": "string", "description": "Integration type identifier"},
    "name": {"type": "string", "description": "Display name"},
    "description": {"type": "string", "description": "Integration description"},
    "icon_url": {"type": "string", "description": "Icon URL"},
    "docs_url": {"type": "string", "description": "Documentation URL"},
    "required_scopes": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Required OAuth scopes",
    },
    "features": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Supported features",
    },
}

_AVAILABLE_INTEGRATIONS_SCHEMA = {
    "integrations": {
        "type": "array",
        "items": {"type": "object", "properties": _AVAILABLE_INTEGRATION_SCHEMA},
        "description": "Available integration types",
    },
}

_INTEGRATION_CONFIG_SCHEMA = {
    "id": {"type": "string", "description": "Integration ID"},
    "type": {"type": "string", "description": "Integration type"},
    "settings": {
        "type": "object",
        "description": "Integration-specific settings",
    },
    "credentials": {
        "type": "object",
        "properties": {
            "has_token": {"type": "boolean", "description": "Whether OAuth token exists"},
            "expires_at": {"type": "string", "format": "date-time", "description": "Token expiry"},
            "scopes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Granted scopes",
            },
        },
        "description": "Credential information (sensitive data masked)",
    },
    "webhooks": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "events": {"type": "array", "items": {"type": "string"}},
            },
        },
        "description": "Configured webhooks",
    },
}

_SYNC_STATUS_SCHEMA = {
    "integration_id": {"type": "string", "description": "Integration ID"},
    "status": {
        "type": "string",
        "enum": ["idle", "syncing", "completed", "failed"],
        "description": "Current sync status",
    },
    "last_sync": {"type": "string", "format": "date-time", "description": "Last sync timestamp"},
    "next_sync": {"type": "string", "format": "date-time", "description": "Next scheduled sync"},
    "items_synced": {"type": "integer", "description": "Items synced in last run"},
    "errors": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Sync errors if any",
    },
}

_TEST_RESULT_SCHEMA = {
    "success": {"type": "boolean", "description": "Whether test passed"},
    "latency_ms": {"type": "number", "description": "Response latency in milliseconds"},
    "message": {"type": "string", "description": "Test result message"},
    "details": {
        "type": "object",
        "description": "Additional test details",
    },
}

_OAUTH_CALLBACK_SCHEMA = {
    "success": {"type": "boolean", "description": "Whether OAuth completed"},
    "integration_id": {"type": "string", "description": "Created/updated integration ID"},
    "message": {"type": "string", "description": "Status message"},
}

_INSTALL_RESULT_SCHEMA = {
    "install_url": {"type": "string", "description": "OAuth authorization URL"},
    "state": {"type": "string", "description": "OAuth state parameter"},
    "expires_in": {"type": "integer", "description": "URL expiry in seconds"},
}

# Connector schemas
_CONNECTOR_SCHEMA = {
    "id": {"type": "string", "description": "Unique connector identifier"},
    "name": {"type": "string", "description": "Connector display name"},
    "type": {
        "type": "string",
        "enum": ["database", "api", "file", "stream", "custom"],
        "description": "Connector type",
    },
    "provider": {
        "type": "string",
        "enum": ["postgres", "mysql", "mongodb", "elasticsearch", "s3", "gcs", "kafka", "custom"],
        "description": "Data provider",
    },
    "status": {
        "type": "string",
        "enum": ["connected", "disconnected", "error", "configuring"],
        "description": "Connection status",
    },
    "config": {
        "type": "object",
        "description": "Connector configuration (credentials masked)",
    },
    "workspace_id": {"type": "string", "description": "Associated workspace"},
    "created_at": {"type": "string", "format": "date-time"},
    "updated_at": {"type": "string", "format": "date-time"},
}

_CONNECTOR_HEALTH_SCHEMA = {
    "connector_id": {"type": "string", "description": "Connector ID"},
    "healthy": {"type": "boolean", "description": "Whether connector is healthy"},
    "latency_ms": {"type": "number", "description": "Connection latency"},
    "last_check": {"type": "string", "format": "date-time", "description": "Last health check"},
    "details": {
        "type": "object",
        "properties": {
            "version": {"type": "string", "description": "Remote system version"},
            "uptime": {"type": "string", "description": "Remote system uptime"},
            "connections_available": {"type": "integer", "description": "Available connections"},
        },
        "description": "Health check details",
    },
}

_CONNECTOR_SYNC_SCHEMA = {
    "sync_id": {"type": "string", "description": "Sync operation ID"},
    "connector_id": {"type": "string", "description": "Connector ID"},
    "status": {
        "type": "string",
        "enum": ["pending", "running", "completed", "failed", "cancelled"],
        "description": "Sync status",
    },
    "started_at": {"type": "string", "format": "date-time"},
    "completed_at": {"type": "string", "format": "date-time"},
    "records_processed": {"type": "integer", "description": "Records processed"},
    "records_failed": {"type": "integer", "description": "Records that failed"},
    "errors": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Error messages",
    },
}

_CONNECTOR_SYNC_LIST_SCHEMA = {
    "syncs": {
        "type": "array",
        "items": {"type": "object", "properties": _CONNECTOR_SYNC_SCHEMA},
        "description": "List of sync operations",
    },
    "total": {"type": "integer", "description": "Total sync count"},
}

# Webhook schemas
_WEBHOOK_SCHEMA = {
    "id": {"type": "string", "description": "Unique webhook identifier"},
    "url": {"type": "string", "description": "Webhook endpoint URL"},
    "events": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Subscribed event types",
    },
    "status": {
        "type": "string",
        "enum": ["active", "paused", "failed"],
        "description": "Webhook status",
    },
    "secret": {"type": "string", "description": "Webhook signing secret (masked)"},
    "created_at": {"type": "string", "format": "date-time"},
    "updated_at": {"type": "string", "format": "date-time"},
}

_WEBHOOK_DELIVERY_SCHEMA = {
    "id": {"type": "string", "description": "Delivery attempt ID"},
    "webhook_id": {"type": "string", "description": "Associated webhook"},
    "event_type": {"type": "string", "description": "Event that triggered delivery"},
    "status": {
        "type": "string",
        "enum": ["pending", "success", "failed", "retrying"],
        "description": "Delivery status",
    },
    "status_code": {"type": "integer", "description": "HTTP response code"},
    "response_time_ms": {"type": "number", "description": "Response time"},
    "attempt_count": {"type": "integer", "description": "Number of attempts"},
    "payload": {"type": "object", "description": "Delivered payload"},
    "response_body": {"type": "string", "description": "Response body (truncated)"},
    "created_at": {"type": "string", "format": "date-time"},
    "delivered_at": {"type": "string", "format": "date-time"},
}

_WEBHOOK_DELIVERY_LIST_SCHEMA = {
    "deliveries": {
        "type": "array",
        "items": {"type": "object", "properties": _WEBHOOK_DELIVERY_SCHEMA},
        "description": "List of delivery attempts",
    },
    "total": {"type": "integer", "description": "Total delivery count"},
    "page": {"type": "integer", "description": "Current page"},
    "page_size": {"type": "integer", "description": "Items per page"},
}

_WEBHOOK_STATS_SCHEMA = {
    "webhook_id": {"type": "string", "description": "Webhook ID"},
    "total_deliveries": {"type": "integer", "description": "Total delivery attempts"},
    "successful_deliveries": {"type": "integer", "description": "Successful deliveries"},
    "failed_deliveries": {"type": "integer", "description": "Failed deliveries"},
    "average_response_time_ms": {"type": "number", "description": "Average response time"},
    "success_rate": {"type": "number", "description": "Success rate (0-1)"},
    "last_delivery": {"type": "string", "format": "date-time", "description": "Last delivery time"},
    "period_start": {"type": "string", "format": "date-time"},
    "period_end": {"type": "string", "format": "date-time"},
}

_RETRY_POLICY_SCHEMA = {
    "max_retries": {"type": "integer", "description": "Maximum retry attempts"},
    "initial_delay_ms": {"type": "integer", "description": "Initial retry delay"},
    "max_delay_ms": {"type": "integer", "description": "Maximum retry delay"},
    "backoff_multiplier": {"type": "number", "description": "Exponential backoff multiplier"},
    "retry_on_status_codes": {
        "type": "array",
        "items": {"type": "integer"},
        "description": "HTTP status codes to retry on",
    },
}

_SIGNING_CONFIG_SCHEMA = {
    "algorithm": {
        "type": "string",
        "enum": ["hmac-sha256", "hmac-sha512", "ed25519"],
        "description": "Signing algorithm",
    },
    "header_name": {"type": "string", "description": "Signature header name"},
    "timestamp_header": {"type": "string", "description": "Timestamp header name"},
    "tolerance_seconds": {"type": "integer", "description": "Timestamp tolerance"},
}

_EVENT_CATEGORY_SCHEMA = {
    "category": {"type": "string", "description": "Category name"},
    "description": {"type": "string", "description": "Category description"},
    "events": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "description": "Event type identifier"},
                "description": {"type": "string", "description": "Event description"},
                "payload_schema": {"type": "object", "description": "Example payload structure"},
            },
        },
        "description": "Events in this category",
    },
}

_EVENT_CATEGORIES_SCHEMA = {
    "categories": {
        "type": "array",
        "items": {"type": "object", "properties": _EVENT_CATEGORY_SCHEMA},
        "description": "Available event categories",
    },
}

_ACTION_RESULT_SCHEMA = {
    "success": {"type": "boolean", "description": "Whether action succeeded"},
    "message": {"type": "string", "description": "Result message"},
    "affected_count": {"type": "integer", "description": "Number of items affected"},
}

_ROTATE_SECRET_SCHEMA = {
    "new_secret": {"type": "string", "description": "New signing secret"},
    "old_secret_valid_until": {
        "type": "string",
        "format": "date-time",
        "description": "When old secret expires",
    },
}

# =============================================================================
# Request Body Schemas
# =============================================================================

_INTEGRATION_CREATE_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": ["slack", "teams", "discord", "email", "zapier", "n8n", "custom"],
            "description": "Integration type",
        },
        "name": {"type": "string", "description": "Display name"},
        "config": {"type": "object", "description": "Integration configuration"},
    },
    "required": ["type", "name"],
}

_INTEGRATION_UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Display name"},
        "config": {"type": "object", "description": "Configuration updates"},
        "status": {
            "type": "string",
            "enum": ["active", "inactive"],
            "description": "Integration status",
        },
    },
}

_OAUTH_CALLBACK_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "OAuth authorization code"},
        "state": {"type": "string", "description": "OAuth state parameter"},
    },
    "required": ["code", "state"],
}

_INSTALL_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "workspace_id": {"type": "string", "description": "Target workspace"},
        "scopes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Requested OAuth scopes",
        },
        "redirect_uri": {"type": "string", "description": "OAuth redirect URI"},
    },
}

_CONNECTOR_CREATE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Connector name"},
        "type": {
            "type": "string",
            "enum": ["database", "api", "file", "stream"],
            "description": "Connector type",
        },
        "provider": {"type": "string", "description": "Data provider"},
        "config": {
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer"},
                "database": {"type": "string"},
                "username": {"type": "string"},
                "password": {"type": "string"},
                "ssl": {"type": "boolean"},
            },
            "description": "Connection configuration",
        },
    },
    "required": ["name", "type", "provider", "config"],
}

_CONNECTOR_UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Connector name"},
        "config": {"type": "object", "description": "Configuration updates"},
    },
}

_SYNC_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "full_sync": {"type": "boolean", "description": "Perform full sync vs incremental"},
        "tables": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Tables to sync (empty = all)",
        },
    },
}

_WEBHOOK_EVENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Event types to subscribe",
        },
    },
    "required": ["events"],
}

_RETRY_POLICY_UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "max_retries": {"type": "integer", "minimum": 0, "maximum": 10},
        "initial_delay_ms": {"type": "integer", "minimum": 100},
        "max_delay_ms": {"type": "integer", "maximum": 3600000},
        "backoff_multiplier": {"type": "number", "minimum": 1.0, "maximum": 10.0},
    },
}

# =============================================================================
# Endpoint Definitions
# =============================================================================

SDK_MISSING_INTEGRATION_ENDPOINTS: dict = {
    # Integrations endpoints
    "/api/integrations": {
        "get": {
            "tags": ["Integrations"],
            "summary": "List all integrations",
            "description": "Get all configured integrations for the workspace.",
            "operationId": "getIntegrations",
            "parameters": [
                {
                    "name": "type",
                    "in": "query",
                    "schema": {"type": "string"},
                    "description": "Filter by type",
                },
                {
                    "name": "status",
                    "in": "query",
                    "schema": {"type": "string"},
                    "description": "Filter by status",
                },
            ],
            "responses": {
                "200": _ok_response("Integrations list", _INTEGRATION_LIST_SCHEMA),
            },
        },
        "post": {
            "tags": ["Integrations"],
            "summary": "Create integration",
            "description": "Create a new integration configuration.",
            "operationId": "postIntegrations",
            "requestBody": {
                "content": {"application/json": {"schema": _INTEGRATION_CREATE_SCHEMA}}
            },
            "responses": {
                "200": _ok_response("Created integration", _INTEGRATION_SCHEMA),
                "400": STANDARD_ERRORS["400"],
            },
        },
    },
    "/api/integrations/available": {
        "get": {
            "tags": ["Integrations"],
            "summary": "List available integrations",
            "description": "Get all available integration types that can be configured.",
            "operationId": "getIntegrationsAvailable",
            "responses": {
                "200": _ok_response("Available integrations", _AVAILABLE_INTEGRATIONS_SCHEMA),
            },
        },
    },
    "/api/integrations/config/{id}": {
        "get": {
            "tags": ["Integrations"],
            "summary": "Get integration configuration",
            "description": "Get detailed configuration for a specific integration.",
            "operationId": "getIntegrationsConfig",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Integration ID",
                }
            ],
            "responses": {
                "200": _ok_response("Integration configuration", _INTEGRATION_CONFIG_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/integrations/discord/callback": {
        "post": {
            "tags": ["Integrations"],
            "summary": "Discord OAuth callback",
            "description": "Handle OAuth callback from Discord authorization.",
            "operationId": "postDiscordCallback",
            "requestBody": {
                "content": {"application/json": {"schema": _OAUTH_CALLBACK_REQUEST_SCHEMA}}
            },
            "responses": {
                "200": _ok_response("OAuth completed", _OAUTH_CALLBACK_SCHEMA),
                "400": STANDARD_ERRORS["400"],
            },
        },
    },
    "/api/integrations/discord/install": {
        "post": {
            "tags": ["Integrations"],
            "summary": "Install Discord integration",
            "description": "Initiate Discord bot installation via OAuth.",
            "operationId": "postDiscordInstall",
            "requestBody": {"content": {"application/json": {"schema": _INSTALL_REQUEST_SCHEMA}}},
            "responses": {
                "200": _ok_response("Install URL generated", _INSTALL_RESULT_SCHEMA),
            },
        },
    },
    "/api/integrations/teams/callback": {
        "post": {
            "tags": ["Integrations"],
            "summary": "Teams OAuth callback",
            "description": "Handle OAuth callback from Microsoft Teams authorization.",
            "operationId": "postTeamsCallback",
            "requestBody": {
                "content": {"application/json": {"schema": _OAUTH_CALLBACK_REQUEST_SCHEMA}}
            },
            "responses": {
                "200": _ok_response("OAuth completed", _OAUTH_CALLBACK_SCHEMA),
                "400": STANDARD_ERRORS["400"],
            },
        },
    },
    "/api/integrations/teams/install": {
        "post": {
            "tags": ["Integrations"],
            "summary": "Install Teams integration",
            "description": "Initiate Microsoft Teams app installation via OAuth.",
            "operationId": "postTeamsInstall",
            "requestBody": {"content": {"application/json": {"schema": _INSTALL_REQUEST_SCHEMA}}},
            "responses": {
                "200": _ok_response("Install URL generated", _INSTALL_RESULT_SCHEMA),
            },
        },
    },
    "/api/integrations/{id}": {
        "delete": {
            "tags": ["Integrations"],
            "summary": "Delete integration",
            "description": "Remove an integration and its configuration.",
            "operationId": "deleteIntegration",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Integration ID",
                }
            ],
            "responses": {
                "200": _ok_response("Integration deleted", _ACTION_RESULT_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "get": {
            "tags": ["Integrations"],
            "summary": "Get integration by ID",
            "description": "Retrieve a specific integration.",
            "operationId": "getIntegrationById",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Integration ID",
                }
            ],
            "responses": {
                "200": _ok_response("Integration details", _INTEGRATION_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "put": {
            "tags": ["Integrations"],
            "summary": "Update integration",
            "description": "Update an existing integration configuration.",
            "operationId": "putIntegration",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Integration ID",
                }
            ],
            "requestBody": {
                "content": {"application/json": {"schema": _INTEGRATION_UPDATE_SCHEMA}}
            },
            "responses": {
                "200": _ok_response("Updated integration", _INTEGRATION_SCHEMA),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/integrations/{id}/sync": {
        "get": {
            "tags": ["Integrations"],
            "summary": "Get sync status",
            "description": "Get the current sync status for an integration.",
            "operationId": "getIntegrationSyncStatus",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Integration ID",
                }
            ],
            "responses": {
                "200": _ok_response("Sync status", _SYNC_STATUS_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "post": {
            "tags": ["Integrations"],
            "summary": "Trigger sync",
            "description": "Trigger a manual sync for an integration.",
            "operationId": "postIntegrationSync",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Integration ID",
                }
            ],
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "full_sync": {
                                    "type": "boolean",
                                    "description": "Perform full sync",
                                },
                            },
                        }
                    }
                }
            },
            "responses": {
                "200": _ok_response("Sync triggered", _SYNC_STATUS_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/integrations/{id}/test": {
        "post": {
            "tags": ["Integrations"],
            "summary": "Test integration",
            "description": "Test connectivity and configuration for an integration.",
            "operationId": "postIntegrationTest",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Integration ID",
                }
            ],
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "timeout_ms": {"type": "integer", "description": "Test timeout"},
                            },
                        }
                    }
                }
            },
            "responses": {
                "200": _ok_response("Test results", _TEST_RESULT_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    # Connectors endpoints
    "/api/v1/connectors": {
        "post": {
            "tags": ["Connectors"],
            "summary": "Create connector",
            "description": "Create a new data connector.",
            "operationId": "postConnector",
            "requestBody": {"content": {"application/json": {"schema": _CONNECTOR_CREATE_SCHEMA}}},
            "responses": {
                "200": _ok_response("Created connector", _CONNECTOR_SCHEMA),
                "400": STANDARD_ERRORS["400"],
            },
        },
    },
    "/api/v1/connectors/{id}": {
        "delete": {
            "tags": ["Connectors"],
            "summary": "Delete connector",
            "description": "Remove a data connector.",
            "operationId": "deleteConnector",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Connector ID",
                }
            ],
            "responses": {
                "200": _ok_response("Connector deleted", _ACTION_RESULT_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "patch": {
            "tags": ["Connectors"],
            "summary": "Update connector",
            "description": "Update connector configuration.",
            "operationId": "patchConnector",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Connector ID",
                }
            ],
            "requestBody": {"content": {"application/json": {"schema": _CONNECTOR_UPDATE_SCHEMA}}},
            "responses": {
                "200": _ok_response("Updated connector", _CONNECTOR_SCHEMA),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/connectors/{id}/health": {
        "get": {
            "tags": ["Connectors"],
            "summary": "Get connector health",
            "description": "Check health status of a connector.",
            "operationId": "getConnectorHealth",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Connector ID",
                }
            ],
            "responses": {
                "200": _ok_response("Health status", _CONNECTOR_HEALTH_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/connectors/{id}/sync": {
        "post": {
            "tags": ["Connectors"],
            "summary": "Trigger connector sync",
            "description": "Start a data sync operation for a connector.",
            "operationId": "postConnectorSync",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Connector ID",
                }
            ],
            "requestBody": {"content": {"application/json": {"schema": _SYNC_REQUEST_SCHEMA}}},
            "responses": {
                "200": _ok_response("Sync started", _CONNECTOR_SYNC_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/connectors/{id}/syncs": {
        "get": {
            "tags": ["Connectors"],
            "summary": "List connector syncs",
            "description": "Get sync history for a connector.",
            "operationId": "getConnectorSyncs",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Connector ID",
                },
                {
                    "name": "status",
                    "in": "query",
                    "schema": {"type": "string"},
                    "description": "Filter by status",
                },
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 20},
                    "description": "Max results",
                },
            ],
            "responses": {
                "200": _ok_response("Sync history", _CONNECTOR_SYNC_LIST_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/connectors/{id}/syncs/{sync_id}": {
        "get": {
            "tags": ["Connectors"],
            "summary": "Get sync by ID",
            "description": "Get details of a specific sync operation.",
            "operationId": "getConnectorSyncById",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Connector ID",
                },
                {
                    "name": "sync_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Sync ID",
                },
            ],
            "responses": {
                "200": _ok_response("Sync details", _CONNECTOR_SYNC_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/connectors/{id}/syncs/{sync_id}/cancel": {
        "post": {
            "tags": ["Connectors"],
            "summary": "Cancel sync",
            "description": "Cancel a running sync operation.",
            "operationId": "postConnectorSyncCancel",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Connector ID",
                },
                {
                    "name": "sync_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Sync ID",
                },
            ],
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "reason": {"type": "string", "description": "Cancellation reason"},
                            },
                        }
                    }
                }
            },
            "responses": {
                "200": _ok_response("Sync cancelled", _CONNECTOR_SYNC_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/connectors/{id}/test": {
        "post": {
            "tags": ["Connectors"],
            "summary": "Test connector",
            "description": "Test connectivity for a connector.",
            "operationId": "postConnectorTest",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Connector ID",
                }
            ],
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "timeout_ms": {"type": "integer", "description": "Test timeout"},
                            },
                        }
                    }
                }
            },
            "responses": {
                "200": _ok_response("Test results", _TEST_RESULT_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    # Webhooks endpoints
    "/api/v1/webhooks/bulk": {
        "delete": {
            "tags": ["Webhooks"],
            "summary": "Bulk delete webhooks",
            "description": "Delete multiple webhooks at once.",
            "operationId": "deleteWebhooksBulk",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "webhook_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "IDs to delete",
                                },
                            },
                            "required": ["webhook_ids"],
                        }
                    }
                }
            },
            "responses": {
                "200": _ok_response("Webhooks deleted", _ACTION_RESULT_SCHEMA),
            },
        },
    },
    "/api/v1/webhooks/events/categories": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "List event categories",
            "description": "Get all available webhook event categories and types.",
            "operationId": "getWebhookEventCategories",
            "responses": {
                "200": _ok_response("Event categories", _EVENT_CATEGORIES_SCHEMA),
            },
        },
    },
    "/api/v1/webhooks/pause-all": {
        "post": {
            "tags": ["Webhooks"],
            "summary": "Pause all webhooks",
            "description": "Pause all webhooks for the workspace.",
            "operationId": "postWebhooksPauseAll",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "reason": {"type": "string", "description": "Pause reason"},
                            },
                        }
                    }
                }
            },
            "responses": {
                "200": _ok_response("Webhooks paused", _ACTION_RESULT_SCHEMA),
            },
        },
    },
    "/api/v1/webhooks/resume-all": {
        "post": {
            "tags": ["Webhooks"],
            "summary": "Resume all webhooks",
            "description": "Resume all paused webhooks for the workspace.",
            "operationId": "postWebhooksResumeAll",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "confirm": {"type": "boolean", "description": "Confirm resume"},
                            },
                        }
                    }
                }
            },
            "responses": {
                "200": _ok_response("Webhooks resumed", _ACTION_RESULT_SCHEMA),
            },
        },
    },
    "/api/v1/webhooks/{id}/deliveries": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "List webhook deliveries",
            "description": "Get delivery history for a webhook.",
            "operationId": "getWebhookDeliveries",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Webhook ID",
                },
                {
                    "name": "status",
                    "in": "query",
                    "schema": {"type": "string"},
                    "description": "Filter by status",
                },
                {"name": "page", "in": "query", "schema": {"type": "integer", "default": 1}},
                {"name": "page_size", "in": "query", "schema": {"type": "integer", "default": 50}},
            ],
            "responses": {
                "200": _ok_response("Delivery history", _WEBHOOK_DELIVERY_LIST_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/webhooks/{id}/deliveries/{delivery_id}": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "Get delivery by ID",
            "description": "Get details of a specific delivery attempt.",
            "operationId": "getWebhookDeliveryById",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Webhook ID",
                },
                {
                    "name": "delivery_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Delivery ID",
                },
            ],
            "responses": {
                "200": _ok_response("Delivery details", _WEBHOOK_DELIVERY_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/webhooks/{id}/deliveries/{delivery_id}/retry": {
        "post": {
            "tags": ["Webhooks"],
            "summary": "Retry delivery",
            "description": "Manually retry a failed delivery.",
            "operationId": "postWebhookDeliveryRetry",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Webhook ID",
                },
                {
                    "name": "delivery_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Delivery ID",
                },
            ],
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "force": {
                                    "type": "boolean",
                                    "description": "Force retry even if succeeded",
                                },
                            },
                        }
                    }
                }
            },
            "responses": {
                "200": _ok_response("Retry initiated", _WEBHOOK_DELIVERY_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/webhooks/{id}/events": {
        "delete": {
            "tags": ["Webhooks"],
            "summary": "Unsubscribe from events",
            "description": "Remove event subscriptions from a webhook.",
            "operationId": "deleteWebhookEvents",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Webhook ID",
                }
            ],
            "requestBody": {"content": {"application/json": {"schema": _WEBHOOK_EVENTS_SCHEMA}}},
            "responses": {
                "200": _ok_response("Events unsubscribed", _WEBHOOK_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "post": {
            "tags": ["Webhooks"],
            "summary": "Subscribe to events",
            "description": "Add event subscriptions to a webhook.",
            "operationId": "postWebhookEvents",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Webhook ID",
                }
            ],
            "requestBody": {"content": {"application/json": {"schema": _WEBHOOK_EVENTS_SCHEMA}}},
            "responses": {
                "200": _ok_response("Events subscribed", _WEBHOOK_SCHEMA),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/webhooks/{id}/retry-policy": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "Get retry policy",
            "description": "Get the retry policy configuration for a webhook.",
            "operationId": "getWebhookRetryPolicy",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Webhook ID",
                }
            ],
            "responses": {
                "200": _ok_response("Retry policy", _RETRY_POLICY_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "put": {
            "tags": ["Webhooks"],
            "summary": "Update retry policy",
            "description": "Update the retry policy for a webhook.",
            "operationId": "putWebhookRetryPolicy",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Webhook ID",
                }
            ],
            "requestBody": {
                "content": {"application/json": {"schema": _RETRY_POLICY_UPDATE_SCHEMA}}
            },
            "responses": {
                "200": _ok_response("Updated retry policy", _RETRY_POLICY_SCHEMA),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/webhooks/{id}/rotate-secret": {
        "post": {
            "tags": ["Webhooks"],
            "summary": "Rotate signing secret",
            "description": "Generate a new signing secret for a webhook.",
            "operationId": "postWebhookRotateSecret",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Webhook ID",
                }
            ],
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "grace_period_hours": {
                                    "type": "integer",
                                    "description": "Hours to keep old secret valid",
                                    "default": 24,
                                },
                            },
                        }
                    }
                }
            },
            "responses": {
                "200": _ok_response("Secret rotated", _ROTATE_SECRET_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/webhooks/{id}/signing": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "Get signing configuration",
            "description": "Get the signature configuration for a webhook.",
            "operationId": "getWebhookSigning",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Webhook ID",
                }
            ],
            "responses": {
                "200": _ok_response("Signing configuration", _SIGNING_CONFIG_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/webhooks/{id}/stats": {
        "get": {
            "tags": ["Webhooks"],
            "summary": "Get webhook statistics",
            "description": "Get delivery statistics for a webhook.",
            "operationId": "getWebhookStats",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Webhook ID",
                },
                {
                    "name": "period",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["day", "week", "month"]},
                    "description": "Statistics period",
                },
            ],
            "responses": {
                "200": _ok_response("Webhook statistics", _WEBHOOK_STATS_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
}

__all__ = ["SDK_MISSING_INTEGRATION_ENDPOINTS"]
