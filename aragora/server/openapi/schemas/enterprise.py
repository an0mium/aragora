"""
Enterprise OpenAPI Schema Definitions.

Schemas for OAuth, workspaces, and retention policies.
"""

from typing import Any

ENTERPRISE_SCHEMAS: dict[str, Any] = {
    "OAuthProvider": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Provider identifier (e.g., 'google', 'github')",
            },
            "name": {"type": "string", "description": "Display name for the provider"},
        },
        "required": ["id", "name"],
    },
    "OAuthProviders": {
        "type": "object",
        "properties": {
            "providers": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/OAuthProvider"},
                "description": "List of available OAuth providers",
            },
        },
        "required": ["providers"],
    },
    "Workspace": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Workspace ID"},
            "organization_id": {"type": "string", "description": "Parent organization ID"},
            "name": {"type": "string", "description": "Workspace name"},
            "created_at": {"type": "string", "format": "date-time"},
            "created_by": {"type": "string"},
            "encrypted": {"type": "boolean", "description": "Whether workspace data is encrypted"},
            "retention_days": {"type": "integer", "description": "Data retention period in days"},
            "sensitivity_level": {"type": "string", "description": "Data sensitivity level"},
            "document_count": {"type": "integer"},
            "storage_bytes": {"type": "integer"},
        },
        "required": ["id", "organization_id", "name"],
    },
    "WorkspaceList": {
        "type": "object",
        "properties": {
            "workspaces": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/Workspace"},
            },
            "total": {"type": "integer"},
        },
        "required": ["workspaces", "total"],
    },
    "RetentionPolicy": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "retention_days": {"type": "integer"},
            "data_types": {"type": "array", "items": {"type": "string"}},
            "enabled": {"type": "boolean"},
            "created_at": {"type": "string", "format": "date-time"},
        },
    },
    "RetentionPolicyList": {
        "type": "object",
        "properties": {
            "policies": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/RetentionPolicy"},
            },
            "total": {"type": "integer"},
        },
    },
}


__all__ = ["ENTERPRISE_SCHEMAS"]
