"""Integration management endpoint definitions."""

from aragora.server.openapi.helpers import STANDARD_ERRORS


# Helper to build inline response
def _response(description: str, schema: dict | None = None) -> dict:
    """Build a response with optional inline schema."""
    resp: dict = {"description": description}
    if schema:
        resp["content"] = {"application/json": {"schema": schema}}
    return resp


INTEGRATION_ENDPOINTS = {
    # OAuth Wizard endpoints
    "/api/v2/integrations/wizard": {
        "get": {
            "tags": ["Integrations", "Wizard"],
            "summary": "Get wizard configuration",
            "description": "Get the complete OAuth wizard configuration including all providers, status, and setup guidance.",
            "operationId": "getWizardConfig",
            "responses": {
                "200": _response(
                    "Wizard configuration",
                    {
                        "type": "object",
                        "properties": {
                            "wizard": {"type": "object"},
                            "generated_at": {"type": "string", "format": "date-time"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v2/integrations/wizard/providers": {
        "get": {
            "tags": ["Integrations", "Wizard"],
            "summary": "List available providers",
            "description": "List all available integration providers with optional filtering.",
            "operationId": "listWizardProviders",
            "parameters": [
                {
                    "name": "category",
                    "in": "query",
                    "description": "Filter by category",
                    "schema": {"type": "string", "enum": ["communication", "development"]},
                },
                {
                    "name": "configured",
                    "in": "query",
                    "description": "Filter by configuration status",
                    "schema": {"type": "boolean"},
                },
            ],
            "responses": {
                "200": _response(
                    "Provider list",
                    {
                        "type": "object",
                        "properties": {
                            "providers": {"type": "array", "items": {"type": "object"}},
                            "total": {"type": "integer"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v2/integrations/wizard/status": {
        "get": {
            "tags": ["Integrations", "Wizard"],
            "summary": "Get all integration statuses",
            "description": "Get detailed status of all integrations including configuration and connection status.",
            "operationId": "getWizardStatus",
            "responses": {
                "200": _response(
                    "Integration statuses",
                    {
                        "type": "object",
                        "properties": {
                            "statuses": {"type": "array", "items": {"type": "object"}},
                            "summary": {"type": "object"},
                            "checked_at": {"type": "string", "format": "date-time"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v2/integrations/wizard/validate": {
        "post": {
            "tags": ["Integrations", "Wizard"],
            "summary": "Validate provider configuration",
            "description": "Validate configuration for a provider before connecting.",
            "operationId": "validateWizardConfig",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["provider"],
                            "properties": {
                                "provider": {"type": "string"},
                                "config": {"type": "object"},
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _response(
                    "Validation results",
                    {
                        "type": "object",
                        "properties": {
                            "provider": {"type": "string"},
                            "valid": {"type": "boolean"},
                            "checks": {"type": "array", "items": {"type": "object"}},
                            "recommendations": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    # Integration management endpoints
    "/api/v2/integrations": {
        "get": {
            "tags": ["Integrations"],
            "summary": "List all integrations",
            "description": "List all platform integrations (Slack, Teams, Discord, Email) for the current tenant.",
            "operationId": "listIntegrations",
            "parameters": [
                {
                    "name": "X-Tenant-ID",
                    "in": "header",
                    "description": "Tenant ID for multi-tenant deployments",
                    "schema": {"type": "string"},
                },
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of results (default: 20, max: 100)",
                    "schema": {"type": "integer", "default": 20, "maximum": 100},
                },
                {
                    "name": "offset",
                    "in": "query",
                    "description": "Pagination offset",
                    "schema": {"type": "integer", "default": 0},
                },
                {
                    "name": "type",
                    "in": "query",
                    "description": "Filter by integration type",
                    "schema": {
                        "type": "string",
                        "enum": ["slack", "teams", "discord", "email"],
                    },
                },
                {
                    "name": "status",
                    "in": "query",
                    "description": "Filter by status",
                    "schema": {"type": "string", "enum": ["active", "inactive"]},
                },
            ],
            "responses": {
                "200": _response(
                    "List of integrations",
                    {
                        "type": "object",
                        "properties": {
                            "integrations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string"},
                                        "workspace_id": {"type": "string"},
                                        "workspace_name": {"type": "string"},
                                        "status": {"type": "string"},
                                        "installed_at": {"type": "number"},
                                    },
                                },
                            },
                            "pagination": {
                                "type": "object",
                                "properties": {
                                    "limit": {"type": "integer"},
                                    "offset": {"type": "integer"},
                                    "total": {"type": "integer"},
                                    "has_more": {"type": "boolean"},
                                },
                            },
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v2/integrations/{type}": {
        "get": {
            "tags": ["Integrations"],
            "summary": "Get integration status",
            "description": "Get the status and details of a specific integration type.",
            "operationId": "getIntegration",
            "parameters": [
                {
                    "name": "type",
                    "in": "path",
                    "required": True,
                    "description": "Integration type",
                    "schema": {
                        "type": "string",
                        "enum": ["slack", "teams", "discord", "email"],
                    },
                },
                {
                    "name": "workspace_id",
                    "in": "query",
                    "description": "Specific workspace/tenant ID to query",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _response(
                    "Integration status",
                    {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "connected": {"type": "boolean"},
                            "workspaces": {"type": "array", "items": {"type": "object"}},
                            "health": {"type": "object"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "delete": {
            "tags": ["Integrations"],
            "summary": "Disconnect integration",
            "description": "Disconnect/deactivate a specific integration workspace.",
            "operationId": "disconnectIntegration",
            "parameters": [
                {
                    "name": "type",
                    "in": "path",
                    "required": True,
                    "description": "Integration type",
                    "schema": {
                        "type": "string",
                        "enum": ["slack", "teams", "discord", "email"],
                    },
                },
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["workspace_id"],
                            "properties": {
                                "workspace_id": {
                                    "type": "string",
                                    "description": "Workspace/tenant ID to disconnect",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _response(
                    "Integration disconnected",
                    {
                        "type": "object",
                        "properties": {
                            "disconnected": {"type": "boolean"},
                            "type": {"type": "string"},
                            "workspace_id": {"type": "string"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v2/integrations/{type}/test": {
        "post": {
            "tags": ["Integrations"],
            "summary": "Test integration connectivity",
            "description": "Test the connectivity and health of a specific integration.",
            "operationId": "testIntegration",
            "parameters": [
                {
                    "name": "type",
                    "in": "path",
                    "required": True,
                    "description": "Integration type",
                    "schema": {
                        "type": "string",
                        "enum": ["slack", "teams", "discord", "email"],
                    },
                },
            ],
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "workspace_id": {
                                    "type": "string",
                                    "description": "Workspace/tenant ID to test",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _response(
                    "Test result",
                    {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "workspace_id": {"type": "string"},
                            "test_result": {
                                "type": "object",
                                "properties": {
                                    "status": {"type": "string"},
                                    "error": {"type": ["string", "null"]},
                                },
                            },
                            "tested_at": {"type": "string", "format": "date-time"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v2/integrations/stats": {
        "get": {
            "tags": ["Integrations"],
            "summary": "Get integration statistics",
            "description": "Get aggregate statistics about all integrations.",
            "operationId": "getIntegrationStats",
            "responses": {
                "200": _response(
                    "Integration statistics",
                    {
                        "type": "object",
                        "properties": {
                            "stats": {
                                "type": "object",
                                "properties": {
                                    "slack": {
                                        "type": "object",
                                        "properties": {
                                            "total_workspaces": {"type": "integer"},
                                            "active_workspaces": {"type": "integer"},
                                        },
                                    },
                                    "teams": {
                                        "type": "object",
                                        "properties": {
                                            "total_workspaces": {"type": "integer"},
                                            "active_workspaces": {"type": "integer"},
                                        },
                                    },
                                    "total_integrations": {"type": "integer"},
                                },
                            },
                            "generated_at": {"type": "string", "format": "date-time"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v2/receipts/{receipt_id}/send-to-channel": {
        "post": {
            "tags": ["Receipts", "Integrations"],
            "summary": "Send receipt to channel",
            "description": "Route a decision receipt to a specific channel (Slack, Teams, Email, Discord).",
            "operationId": "sendReceiptToChannel",
            "parameters": [
                {
                    "name": "receipt_id",
                    "in": "path",
                    "required": True,
                    "description": "Receipt ID",
                    "schema": {"type": "string"},
                },
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["channel_type", "channel_id"],
                            "properties": {
                                "channel_type": {
                                    "type": "string",
                                    "enum": ["slack", "teams", "email", "discord"],
                                    "description": "Target channel type",
                                },
                                "channel_id": {
                                    "type": "string",
                                    "description": "Channel/conversation ID or email address",
                                },
                                "workspace_id": {
                                    "type": "string",
                                    "description": "Workspace/tenant ID (required for Slack/Teams)",
                                },
                                "options": {
                                    "type": "object",
                                    "properties": {
                                        "compact": {
                                            "type": "boolean",
                                            "default": False,
                                            "description": "Use compact formatting",
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _response(
                    "Receipt sent successfully",
                    {
                        "type": "object",
                        "properties": {
                            "sent": {"type": "boolean"},
                            "receipt_id": {"type": "string"},
                            "channel_type": {"type": "string"},
                            "channel_id": {"type": "string"},
                            "message_ts": {"type": "string"},
                            "message_id": {"type": "string"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
                "501": {
                    "description": "Channel not available",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Error"},
                        },
                    },
                },
            },
        },
    },
    "/api/v2/receipts/{receipt_id}/formatted/{channel_type}": {
        "get": {
            "tags": ["Receipts", "Integrations"],
            "summary": "Get formatted receipt",
            "description": "Get a receipt formatted for a specific channel type without sending it.",
            "operationId": "getFormattedReceipt",
            "parameters": [
                {
                    "name": "receipt_id",
                    "in": "path",
                    "required": True,
                    "description": "Receipt ID",
                    "schema": {"type": "string"},
                },
                {
                    "name": "channel_type",
                    "in": "path",
                    "required": True,
                    "description": "Channel type for formatting",
                    "schema": {
                        "type": "string",
                        "enum": ["slack", "teams", "email", "discord"],
                    },
                },
                {
                    "name": "compact",
                    "in": "query",
                    "description": "Use compact formatting",
                    "schema": {"type": "boolean", "default": False},
                },
            ],
            "responses": {
                "200": _response(
                    "Formatted receipt",
                    {
                        "type": "object",
                        "properties": {
                            "receipt_id": {"type": "string"},
                            "channel_type": {"type": "string"},
                            "formatted": {
                                "type": "object",
                                "description": "Channel-specific formatted content",
                            },
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
}
