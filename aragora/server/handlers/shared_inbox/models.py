"""
Data models for Shared Inbox.

Contains dataclasses and enums for shared inbox, messages, and routing rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MessageStatus(str, Enum):
    """Status of a message in the shared inbox."""

    OPEN = "open"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    WAITING = "waiting"
    RESOLVED = "resolved"
    CLOSED = "closed"


class RuleConditionField(str, Enum):
    """Fields that can be used in routing rule conditions."""

    FROM = "from"
    TO = "to"
    SUBJECT = "subject"
    BODY = "body"
    LABELS = "labels"
    PRIORITY = "priority"
    SENDER_DOMAIN = "sender_domain"


class RuleConditionOperator(str, Enum):
    """Operators for routing rule conditions."""

    CONTAINS = "contains"
    EQUALS = "equals"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES = "matches"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"


class RuleActionType(str, Enum):
    """Actions that routing rules can perform."""

    ASSIGN = "assign"
    LABEL = "label"
    ESCALATE = "escalate"
    ARCHIVE = "archive"
    NOTIFY = "notify"
    FORWARD = "forward"


@dataclass
class RuleCondition:
    """A single condition in a routing rule."""

    field: RuleConditionField
    operator: RuleConditionOperator
    value: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field.value,
            "operator": self.operator.value,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuleCondition":
        return cls(
            field=RuleConditionField(data["field"]),
            operator=RuleConditionOperator(data["operator"]),
            value=data["value"],
        )


@dataclass
class RuleAction:
    """An action to perform when a routing rule matches."""

    type: RuleActionType
    target: str | None = None
    params: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "target": self.target,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuleAction":
        return cls(
            type=RuleActionType(data["type"]),
            target=data.get("target"),
            params=data.get("params", {}),
        )


@dataclass
class RoutingRule:
    """A rule for automatically routing emails."""

    id: str
    name: str
    workspace_id: str
    conditions: list[RuleCondition]
    condition_logic: str  # "AND" or "OR"
    actions: list[RuleAction]
    priority: int = 5
    enabled: bool = True
    description: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "workspace_id": self.workspace_id,
            "conditions": [c.to_dict() for c in self.conditions],
            "condition_logic": self.condition_logic,
            "actions": [a.to_dict() for a in self.actions],
            "priority": self.priority,
            "enabled": self.enabled,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoutingRule":
        return cls(
            id=data["id"],
            name=data["name"],
            workspace_id=data["workspace_id"],
            conditions=[RuleCondition.from_dict(c) for c in data.get("conditions", [])],
            condition_logic=data.get("condition_logic", "AND"),
            actions=[RuleAction.from_dict(a) for a in data.get("actions", [])],
            priority=data.get("priority", 5),
            enabled=data.get("enabled", True),
            description=data.get("description"),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else datetime.now(timezone.utc)
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if data.get("updated_at")
                else datetime.now(timezone.utc)
            ),
            created_by=data.get("created_by"),
            stats=data.get("stats", {}),
        )


@dataclass
class SharedInboxMessage:
    """A message in a shared inbox with collaboration metadata."""

    id: str
    inbox_id: str
    email_id: str  # Original email ID from connector
    subject: str
    from_address: str
    to_addresses: list[str]
    snippet: str
    received_at: datetime
    status: MessageStatus = MessageStatus.OPEN
    assigned_to: str | None = None
    assigned_at: datetime | None = None
    tags: list[str] = field(default_factory=list)
    priority: str | None = None
    notes: list[dict[str, Any]] = field(default_factory=list)
    thread_id: str | None = None
    sla_deadline: datetime | None = None
    resolved_at: datetime | None = None
    resolved_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "inbox_id": self.inbox_id,
            "email_id": self.email_id,
            "subject": self.subject,
            "from_address": self.from_address,
            "to_addresses": self.to_addresses,
            "snippet": self.snippet,
            "received_at": self.received_at.isoformat(),
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "tags": self.tags,
            "priority": self.priority,
            "notes": self.notes,
            "thread_id": self.thread_id,
            "sla_deadline": self.sla_deadline.isoformat() if self.sla_deadline else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
        }


@dataclass
class SharedInbox:
    """A shared inbox for team collaboration."""

    id: str
    workspace_id: str
    name: str
    description: str | None = None
    email_address: str | None = None  # Associated email address
    connector_type: str | None = None  # "gmail", "outlook", etc.
    team_members: list[str] = field(default_factory=list)
    admins: list[str] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str | None = None
    message_count: int = 0
    unread_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "name": self.name,
            "description": self.description,
            "email_address": self.email_address,
            "connector_type": self.connector_type,
            "team_members": self.team_members,
            "admins": self.admins,
            "settings": self.settings,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "message_count": self.message_count,
            "unread_count": self.unread_count,
        }
