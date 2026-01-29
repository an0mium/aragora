"""
Unified Inbox OpenAPI Schema Definitions.

Schemas for shared inboxes and routing rules.
"""

from typing import Any

INBOX_SCHEMAS: dict[str, Any] = {
    "SharedInbox": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "workspace_id": {"type": "string"},
            "name": {"type": "string"},
            "description": {"type": "string", "nullable": True},
            "email_address": {"type": "string", "nullable": True},
            "connector_type": {"type": "string", "nullable": True},
            "team_members": {"type": "array", "items": {"type": "string"}},
            "admins": {"type": "array", "items": {"type": "string"}},
            "message_count": {"type": "integer"},
            "unread_count": {"type": "integer"},
            "settings": {"type": "object"},
            "created_at": {"type": "string"},
            "updated_at": {"type": "string"},
            "created_by": {"type": "string", "nullable": True},
        },
    },
    "SharedInboxMessage": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "inbox_id": {"type": "string"},
            "email_id": {"type": "string"},
            "subject": {"type": "string"},
            "from_address": {"type": "string"},
            "to_addresses": {"type": "array", "items": {"type": "string"}},
            "snippet": {"type": "string"},
            "received_at": {"type": "string"},
            "status": {"type": "string"},
            "assigned_to": {"type": "string", "nullable": True},
            "tags": {"type": "array", "items": {"type": "string"}},
            "priority": {"type": "string", "nullable": True},
        },
    },
    "SharedInboxResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "inbox": {"$ref": "#/components/schemas/SharedInbox"},
            "error": {"type": "string", "nullable": True},
        },
        "required": ["success"],
    },
    "SharedInboxListResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "inboxes": {"type": "array", "items": {"$ref": "#/components/schemas/SharedInbox"}},
            "total": {"type": "integer"},
        },
        "required": ["success", "inboxes", "total"],
    },
    "SharedInboxMessageListResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "messages": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/SharedInboxMessage"},
            },
            "total": {"type": "integer"},
            "limit": {"type": "integer"},
            "offset": {"type": "integer"},
        },
        "required": ["success", "messages", "total"],
    },
    "SharedInboxMessageResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "message": {"$ref": "#/components/schemas/SharedInboxMessage"},
            "error": {"type": "string", "nullable": True},
        },
        "required": ["success"],
    },
    "SharedInboxCreateRequest": {
        "type": "object",
        "properties": {
            "workspace_id": {"type": "string"},
            "name": {"type": "string"},
            "description": {"type": "string"},
            "email_address": {"type": "string"},
            "connector_type": {"type": "string"},
            "team_members": {"type": "array", "items": {"type": "string"}},
            "admins": {"type": "array", "items": {"type": "string"}},
            "settings": {"type": "object"},
        },
        "required": ["workspace_id", "name"],
    },
    "SharedInboxAssignRequest": {
        "type": "object",
        "properties": {
            "assigned_to": {"type": "string"},
        },
        "required": ["assigned_to"],
    },
    "SharedInboxStatusRequest": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
        },
        "required": ["status"],
    },
    "SharedInboxTagRequest": {
        "type": "object",
        "properties": {
            "tag": {"type": "string"},
        },
        "required": ["tag"],
    },
    "RoutingRule": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "workspace_id": {"type": "string"},
            "name": {"type": "string"},
            "conditions": {"type": "array", "items": {"type": "object"}},
            "condition_logic": {"type": "string"},
            "actions": {"type": "array", "items": {"type": "object"}},
            "priority": {"type": "integer"},
            "enabled": {"type": "boolean"},
            "description": {"type": "string", "nullable": True},
            "created_at": {"type": "string"},
            "updated_at": {"type": "string"},
            "created_by": {"type": "string", "nullable": True},
            "stats": {"type": "object"},
        },
    },
    "RoutingRuleResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "rule": {"$ref": "#/components/schemas/RoutingRule"},
            "error": {"type": "string", "nullable": True},
        },
        "required": ["success"],
    },
    "RoutingRuleListResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "rules": {"type": "array", "items": {"$ref": "#/components/schemas/RoutingRule"}},
            "total": {"type": "integer"},
            "limit": {"type": "integer"},
            "offset": {"type": "integer"},
        },
        "required": ["success", "rules", "total"],
    },
    "RoutingRuleDeleteResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "deleted": {"type": "string"},
        },
        "required": ["success"],
    },
    "RoutingRuleTestResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "rule_id": {"type": "string"},
            "match_count": {"type": "integer"},
            "rule": {"$ref": "#/components/schemas/RoutingRule"},
            "error": {"type": "string", "nullable": True},
        },
        "required": ["success"],
    },
    "RoutingRuleCreateRequest": {
        "type": "object",
        "properties": {
            "workspace_id": {"type": "string"},
            "name": {"type": "string"},
            "conditions": {"type": "array", "items": {"type": "object"}},
            "condition_logic": {"type": "string"},
            "actions": {"type": "array", "items": {"type": "object"}},
            "priority": {"type": "integer"},
            "enabled": {"type": "boolean"},
            "description": {"type": "string"},
        },
        "required": ["workspace_id", "name", "conditions", "actions"],
    },
    "RoutingRuleUpdateRequest": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "conditions": {"type": "array", "items": {"type": "object"}},
            "condition_logic": {"type": "string"},
            "actions": {"type": "array", "items": {"type": "object"}},
            "priority": {"type": "integer"},
            "enabled": {"type": "boolean"},
        },
    },
    "RoutingRuleTestRequest": {
        "type": "object",
        "properties": {
            "workspace_id": {"type": "string"},
        },
        "required": ["workspace_id"],
    },
}


__all__ = ["INBOX_SCHEMAS"]
