"""Gateway endpoint definitions for IoT device gateway management."""

from typing import Any

from aragora.server.openapi.helpers import STANDARD_ERRORS


def _response(description: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a response with optional inline schema."""
    resp: dict[str, Any] = {"description": description}
    if schema:
        resp["content"] = {"application/json": {"schema": schema}}
    return resp


# ---- Reusable schemas -------------------------------------------------------

_DEVICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "device_id": {"type": "string", "description": "Unique device identifier"},
        "name": {"type": "string", "description": "Human-readable device name"},
        "device_type": {"type": "string", "description": "Device type classification"},
        "capabilities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of device capabilities",
        },
        "status": {
            "type": "string",
            "enum": ["online", "offline", "degraded", "unknown"],
            "description": "Current device status",
        },
        "paired_at": {
            "type": "string",
            "format": "date-time",
            "description": "Timestamp when device was registered",
        },
        "last_seen": {
            "type": "string",
            "format": "date-time",
            "description": "Timestamp of last device heartbeat",
        },
        "allowed_channels": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Channels the device is allowed to communicate on",
        },
        "metadata": {
            "type": "object",
            "description": "Arbitrary key-value metadata",
        },
    },
}

_CHANNEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Channel name"},
        "status": {
            "type": "string",
            "enum": ["available", "unavailable"],
            "description": "Channel availability status",
        },
    },
}

_MESSAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "message_id": {"type": "string", "description": "Unique message identifier"},
        "channel": {"type": "string", "description": "Channel the message was routed through"},
        "content": {"type": "string", "description": "Message content"},
        "agent_id": {"type": "string", "description": "ID of the agent that handled the message"},
        "rule_id": {"type": "string", "description": "ID of the routing rule that matched"},
        "routed_at": {
            "type": "string",
            "format": "date-time",
            "description": "Timestamp when message was routed",
        },
        "status": {
            "type": "string",
            "enum": ["pending", "routed", "delivered", "failed"],
            "description": "Message delivery status",
        },
    },
}

_ROUTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "Unique route rule identifier"},
        "channel": {"type": "string", "description": "Target channel for routing"},
        "pattern": {"type": "string", "description": "Message pattern to match"},
        "agent_id": {"type": "string", "description": "Agent to route matched messages to"},
        "priority": {
            "type": "integer",
            "description": "Rule evaluation priority (lower is higher)",
        },
        "enabled": {"type": "boolean", "description": "Whether the rule is active"},
    },
}

# ---- Path parameter helpers -------------------------------------------------

_DEVICE_ID_PARAM: dict[str, Any] = {
    "name": "device_id",
    "in": "path",
    "required": True,
    "description": "Unique device identifier",
    "schema": {"type": "string"},
}

_MESSAGE_ID_PARAM: dict[str, Any] = {
    "name": "message_id",
    "in": "path",
    "required": True,
    "description": "Unique message identifier",
    "schema": {"type": "string"},
}

_ROUTE_ID_PARAM: dict[str, Any] = {
    "name": "route_id",
    "in": "path",
    "required": True,
    "description": "Unique routing rule identifier",
    "schema": {"type": "string"},
}

# =============================================================================
# Gateway endpoint definitions
# =============================================================================

GATEWAY_ENDPOINTS: dict[str, Any] = {
    # -------------------------------------------------------------------------
    # Channels
    # -------------------------------------------------------------------------
    "/api/v1/gateway/channels": {
        "get": {
            "tags": ["Gateway"],
            "summary": "List gateway channels",
            "description": "List all available gateway channels and their current status.",
            "operationId": "listGatewayChannels",
            "responses": {
                "200": _response(
                    "List of gateway channels",
                    {
                        "type": "object",
                        "properties": {
                            "channels": {
                                "type": "array",
                                "items": _CHANNEL_SCHEMA,
                            },
                            "total": {"type": "integer"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "post": {
            "tags": ["Gateway"],
            "summary": "Create gateway channel",
            "description": "Register a new gateway channel for device communication.",
            "operationId": "createGatewayChannel",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["name"],
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Channel name",
                                },
                                "protocol": {
                                    "type": "string",
                                    "description": "Communication protocol (e.g. mqtt, http, websocket)",
                                },
                                "metadata": {
                                    "type": "object",
                                    "description": "Additional channel configuration",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "201": _response(
                    "Channel created",
                    {
                        "type": "object",
                        "properties": {
                            "channel": _CHANNEL_SCHEMA,
                            "message": {"type": "string"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    # -------------------------------------------------------------------------
    # Devices (collection)
    # -------------------------------------------------------------------------
    "/api/v1/gateway/devices": {
        "get": {
            "tags": ["Gateway"],
            "summary": "List registered devices",
            "description": "List all devices registered with the gateway. Supports filtering by status and device type.",
            "operationId": "listGatewayDevices",
            "parameters": [
                {
                    "name": "status",
                    "in": "query",
                    "description": "Filter devices by status",
                    "schema": {
                        "type": "string",
                        "enum": ["online", "offline", "degraded", "unknown"],
                    },
                },
                {
                    "name": "type",
                    "in": "query",
                    "description": "Filter devices by type",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _response(
                    "List of registered devices",
                    {
                        "type": "object",
                        "properties": {
                            "devices": {
                                "type": "array",
                                "items": _DEVICE_SCHEMA,
                            },
                            "total": {"type": "integer"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "post": {
            "tags": ["Gateway"],
            "summary": "Register a device",
            "description": "Register a new device with the gateway. The device name is required; a device_id will be generated if not supplied.",
            "operationId": "registerGatewayDevice",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["name"],
                            "properties": {
                                "device_id": {
                                    "type": "string",
                                    "description": "Optional explicit device ID",
                                },
                                "name": {
                                    "type": "string",
                                    "description": "Human-readable device name",
                                },
                                "device_type": {
                                    "type": "string",
                                    "description": "Device type classification",
                                    "default": "unknown",
                                },
                                "capabilities": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Device capabilities",
                                },
                                "allowed_channels": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Channels the device may communicate on",
                                },
                                "metadata": {
                                    "type": "object",
                                    "description": "Arbitrary key-value metadata",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "201": _response(
                    "Device registered",
                    {
                        "type": "object",
                        "properties": {
                            "device_id": {"type": "string"},
                            "message": {"type": "string"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    # -------------------------------------------------------------------------
    # Devices (individual)
    # -------------------------------------------------------------------------
    "/api/v1/gateway/devices/{device_id}": {
        "get": {
            "tags": ["Gateway"],
            "summary": "Get device details",
            "description": "Retrieve full details for a specific registered device, including allowed channels and metadata.",
            "operationId": "getGatewayDevice",
            "parameters": [_DEVICE_ID_PARAM],
            "responses": {
                "200": _response(
                    "Device details",
                    {
                        "type": "object",
                        "properties": {
                            "device": _DEVICE_SCHEMA,
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "put": {
            "tags": ["Gateway"],
            "summary": "Update device",
            "description": "Update the configuration of an existing registered device.",
            "operationId": "updateGatewayDevice",
            "parameters": [_DEVICE_ID_PARAM],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Updated device name",
                                },
                                "device_type": {
                                    "type": "string",
                                    "description": "Updated device type",
                                },
                                "capabilities": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Updated capabilities list",
                                },
                                "allowed_channels": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Updated allowed channels",
                                },
                                "metadata": {
                                    "type": "object",
                                    "description": "Updated metadata",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _response(
                    "Device updated",
                    {
                        "type": "object",
                        "properties": {
                            "device": _DEVICE_SCHEMA,
                            "message": {"type": "string"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "delete": {
            "tags": ["Gateway"],
            "summary": "Unregister device",
            "description": "Remove a device from the gateway registry. This is irreversible.",
            "operationId": "deleteGatewayDevice",
            "parameters": [_DEVICE_ID_PARAM],
            "responses": {
                "200": _response(
                    "Device unregistered",
                    {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    # -------------------------------------------------------------------------
    # Messages (collection)
    # -------------------------------------------------------------------------
    "/api/v1/gateway/messages": {
        "get": {
            "tags": ["Gateway"],
            "summary": "List routed messages",
            "description": "List messages that have been routed through the gateway. Supports filtering by channel and status.",
            "operationId": "listGatewayMessages",
            "parameters": [
                {
                    "name": "channel",
                    "in": "query",
                    "description": "Filter by channel name",
                    "schema": {"type": "string"},
                },
                {
                    "name": "status",
                    "in": "query",
                    "description": "Filter by delivery status",
                    "schema": {
                        "type": "string",
                        "enum": ["pending", "routed", "delivered", "failed"],
                    },
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
            ],
            "responses": {
                "200": _response(
                    "List of routed messages",
                    {
                        "type": "object",
                        "properties": {
                            "messages": {
                                "type": "array",
                                "items": _MESSAGE_SCHEMA,
                            },
                            "total": {"type": "integer"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "post": {
            "tags": ["Gateway"],
            "summary": "Route a message",
            "description": "Submit a message to the gateway for routing to the appropriate agent based on configured rules.",
            "operationId": "routeGatewayMessage",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["channel", "content"],
                            "properties": {
                                "channel": {
                                    "type": "string",
                                    "description": "Target channel for routing",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Message content to route",
                                },
                                "metadata": {
                                    "type": "object",
                                    "description": "Additional message metadata",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _response(
                    "Message routed",
                    {
                        "type": "object",
                        "properties": {
                            "routed": {"type": "boolean"},
                            "agent_id": {
                                "type": "string",
                                "description": "Agent the message was routed to",
                            },
                            "rule_id": {
                                "type": "string",
                                "description": "Routing rule that matched",
                            },
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    # -------------------------------------------------------------------------
    # Messages (individual)
    # -------------------------------------------------------------------------
    "/api/v1/gateway/messages/{message_id}": {
        "get": {
            "tags": ["Gateway"],
            "summary": "Get message details",
            "description": "Retrieve the full details and routing information for a specific message.",
            "operationId": "getGatewayMessage",
            "parameters": [_MESSAGE_ID_PARAM],
            "responses": {
                "200": _response(
                    "Message details",
                    {
                        "type": "object",
                        "properties": {
                            "message": _MESSAGE_SCHEMA,
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "delete": {
            "tags": ["Gateway"],
            "summary": "Delete a message",
            "description": "Delete a routed message from the gateway message log.",
            "operationId": "deleteGatewayMessage",
            "parameters": [_MESSAGE_ID_PARAM],
            "responses": {
                "200": _response(
                    "Message deleted",
                    {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    # -------------------------------------------------------------------------
    # Routing rules (collection)
    # -------------------------------------------------------------------------
    "/api/v1/gateway/routing": {
        "get": {
            "tags": ["Gateway"],
            "summary": "List routing rules",
            "description": "List all routing rules configured in the gateway, including statistics.",
            "operationId": "listGatewayRoutingRules",
            "responses": {
                "200": _response(
                    "List of routing rules",
                    {
                        "type": "object",
                        "properties": {
                            "rules": {
                                "type": "array",
                                "items": _ROUTE_SCHEMA,
                            },
                            "total": {"type": "integer"},
                            "stats": {
                                "type": "object",
                                "properties": {
                                    "total_rules": {"type": "integer"},
                                    "messages_routed": {"type": "integer"},
                                    "routing_errors": {"type": "integer"},
                                },
                                "description": "Aggregate routing statistics",
                            },
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "post": {
            "tags": ["Gateway"],
            "summary": "Create routing rule",
            "description": "Create a new routing rule that maps message patterns on a channel to a specific agent.",
            "operationId": "createGatewayRoutingRule",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["channel", "pattern", "agent_id"],
                            "properties": {
                                "channel": {
                                    "type": "string",
                                    "description": "Channel to apply the rule to",
                                },
                                "pattern": {
                                    "type": "string",
                                    "description": "Message matching pattern (regex supported)",
                                },
                                "agent_id": {
                                    "type": "string",
                                    "description": "Target agent for matched messages",
                                },
                                "priority": {
                                    "type": "integer",
                                    "description": "Rule evaluation priority (lower is higher)",
                                    "default": 100,
                                },
                                "enabled": {
                                    "type": "boolean",
                                    "description": "Whether the rule is active",
                                    "default": True,
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "201": _response(
                    "Routing rule created",
                    {
                        "type": "object",
                        "properties": {
                            "rule": _ROUTE_SCHEMA,
                            "message": {"type": "string"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    # -------------------------------------------------------------------------
    # Routing rules (individual)
    # -------------------------------------------------------------------------
    "/api/v1/gateway/routing/{route_id}": {
        "get": {
            "tags": ["Gateway"],
            "summary": "Get routing rule",
            "description": "Retrieve details for a specific routing rule.",
            "operationId": "getGatewayRoutingRule",
            "parameters": [_ROUTE_ID_PARAM],
            "responses": {
                "200": _response(
                    "Routing rule details",
                    {
                        "type": "object",
                        "properties": {
                            "rule": _ROUTE_SCHEMA,
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "put": {
            "tags": ["Gateway"],
            "summary": "Update routing rule",
            "description": "Update an existing routing rule configuration.",
            "operationId": "updateGatewayRoutingRule",
            "parameters": [_ROUTE_ID_PARAM],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "channel": {
                                    "type": "string",
                                    "description": "Updated target channel",
                                },
                                "pattern": {
                                    "type": "string",
                                    "description": "Updated message matching pattern",
                                },
                                "agent_id": {
                                    "type": "string",
                                    "description": "Updated target agent",
                                },
                                "priority": {
                                    "type": "integer",
                                    "description": "Updated priority",
                                },
                                "enabled": {
                                    "type": "boolean",
                                    "description": "Updated active status",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _response(
                    "Routing rule updated",
                    {
                        "type": "object",
                        "properties": {
                            "rule": _ROUTE_SCHEMA,
                            "message": {"type": "string"},
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "delete": {
            "tags": ["Gateway"],
            "summary": "Delete routing rule",
            "description": "Remove a routing rule from the gateway. Messages will no longer match this rule.",
            "operationId": "deleteGatewayRoutingRule",
            "parameters": [_ROUTE_ID_PARAM],
            "responses": {
                "200": _response(
                    "Routing rule deleted",
                    {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"},
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
}
