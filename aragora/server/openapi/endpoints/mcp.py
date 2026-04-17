"""Manual OpenAPI definitions for MCP tool discovery routes."""

from __future__ import annotations

from typing import Any

from aragora.server.openapi.helpers import STANDARD_ERRORS, _ok_response


_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["name", "description", "parameters"],
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "parameters": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "required": {"type": "boolean"},
                    "default": {},
                },
            },
        },
    },
}


MCP_ENDPOINTS = {
    "/api/v1/mcp/tools": {
        "get": {
            "tags": ["MCP"],
            "summary": "List MCP tools",
            "description": (
                "Return the Aragora MCP tool catalog sourced from "
                "`aragora.mcp.tools.TOOLS_METADATA`. Supports optional prefix "
                "filtering with the `category` query parameter."
            ),
            "operationId": "listMcpToolsV1",
            "parameters": [
                {
                    "name": "category",
                    "in": "query",
                    "required": False,
                    "description": "Optional tool-name prefix filter.",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "MCP tool catalog",
                    {
                        "tools": {"type": "array", "items": _TOOL_SCHEMA},
                        "count": {"type": "integer"},
                    },
                ),
                "500": STANDARD_ERRORS["500"],
                "503": {"description": "MCP tools module not available"},
            },
        }
    },
    "/api/v1/mcp/tools/{name}": {
        "get": {
            "tags": ["MCP"],
            "summary": "Get MCP tool",
            "description": "Return metadata for a single Aragora MCP tool by name.",
            "operationId": "getMcpToolV1",
            "parameters": [
                {
                    "name": "name",
                    "in": "path",
                    "required": True,
                    "description": "MCP tool name.",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("MCP tool details", {"tool": _TOOL_SCHEMA}),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
                "503": {"description": "MCP tools module not available"},
            },
        }
    },
}
