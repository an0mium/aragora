"""
HTTP API Handlers for Shared Inbox Management.

Provides REST APIs for collaborative email inbox management:
- Create and manage shared inboxes
- Message assignment and ownership
- Team-based inbox views
- Status tracking (open, assigned, resolved)
- Tagging and labeling

Endpoints:
- POST /api/v1/inbox/shared - Create shared inbox
- GET /api/v1/inbox/shared - List shared inboxes
- GET /api/v1/inbox/shared/:id - Get shared inbox details
- GET /api/v1/inbox/shared/:id/messages - Get messages in inbox
- POST /api/v1/inbox/shared/:id/messages/:msg_id/assign - Assign message
- POST /api/v1/inbox/shared/:id/messages/:msg_id/status - Update status
- POST /api/v1/inbox/shared/:id/messages/:msg_id/tag - Add tag
- POST /api/v1/inbox/routing/rules - Create routing rule
- GET /api/v1/inbox/routing/rules - List routing rules
- PATCH /api/v1/inbox/routing/rules/:id - Update routing rule
- DELETE /api/v1/inbox/routing/rules/:id - Delete routing rule
- POST /api/v1/inbox/routing/rules/:id/test - Test routing rule
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Persistent Storage Access
# =============================================================================

_email_store = None
_email_store_lock = threading.Lock()

_rules_store = None
_rules_store_lock = threading.Lock()


def _get_email_store():
    """Get or create the email store (lazy init, thread-safe)."""
    global _email_store
    if _email_store is not None:
        return _email_store
    with _email_store_lock:
        if _email_store is None:
            try:
                from aragora.storage.email_store import get_email_store

                _email_store = get_email_store()
                logger.info("[SharedInbox] Initialized persistent email store")
            except Exception as e:
                logger.warning(f"[SharedInbox] Failed to init email store: {e}")
        return _email_store


def _get_rules_store():
    """Get or create the rules store (lazy init, thread-safe)."""
    global _rules_store
    if _rules_store is not None:
        return _rules_store
    with _rules_store_lock:
        if _rules_store is None:
            try:
                from aragora.services.rules_store import get_rules_store

                _rules_store = get_rules_store()
                logger.info("[SharedInbox] Initialized persistent rules store")
            except Exception as e:
                logger.warning(f"[SharedInbox] Failed to init rules store: {e}")
        return _rules_store


# Storage configuration
USE_PERSISTENT_STORAGE = True  # Set to False for in-memory only (testing)


# =============================================================================
# Data Models
# =============================================================================


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field.value,
            "operator": self.operator.value,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuleCondition":
        return cls(
            field=RuleConditionField(data["field"]),
            operator=RuleConditionOperator(data["operator"]),
            value=data["value"],
        )


@dataclass
class RuleAction:
    """An action to perform when a routing rule matches."""

    type: RuleActionType
    target: Optional[str] = None
    params: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "target": self.target,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuleAction":
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
    conditions: List[RuleCondition]
    condition_logic: str  # "AND" or "OR"
    actions: List[RuleAction]
    priority: int = 5
    enabled: bool = True
    description: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingRule":
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
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else datetime.now(timezone.utc),
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
    to_addresses: List[str]
    snippet: str
    received_at: datetime
    status: MessageStatus = MessageStatus.OPEN
    assigned_to: Optional[str] = None
    assigned_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    priority: Optional[str] = None
    notes: List[Dict[str, Any]] = field(default_factory=list)
    thread_id: Optional[str] = None
    sla_deadline: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
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
    description: Optional[str] = None
    email_address: Optional[str] = None  # Associated email address
    connector_type: Optional[str] = None  # "gmail", "outlook", etc.
    team_members: List[str] = field(default_factory=list)
    admins: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    message_count: int = 0
    unread_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
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


# =============================================================================
# In-Memory Storage (fallback when USE_PERSISTENT_STORAGE=False)
# =============================================================================

_shared_inboxes: Dict[str, SharedInbox] = {}
_inbox_messages: Dict[str, Dict[str, SharedInboxMessage]] = {}  # inbox_id -> {msg_id -> message}
_routing_rules: Dict[str, RoutingRule] = {}
_storage_lock = threading.Lock()


def _get_store():
    """Get the persistent storage instance if enabled."""
    if not USE_PERSISTENT_STORAGE:
        return None
    return _get_email_store()


# =============================================================================
# Shared Inbox Handlers
# =============================================================================


async def handle_create_shared_inbox(
    workspace_id: str,
    name: str,
    description: Optional[str] = None,
    email_address: Optional[str] = None,
    connector_type: Optional[str] = None,
    team_members: Optional[List[str]] = None,
    admins: Optional[List[str]] = None,
    settings: Optional[Dict[str, Any]] = None,
    created_by: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new shared inbox.

    POST /api/v1/inbox/shared
    {
        "workspace_id": "ws_123",
        "name": "Support Inbox",
        "description": "Customer support emails",
        "email_address": "support@company.com",
        "connector_type": "gmail",
        "team_members": ["user1", "user2"],
        "admins": ["admin1"]
    }
    """
    try:
        inbox_id = f"inbox_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        inbox = SharedInbox(
            id=inbox_id,
            workspace_id=workspace_id,
            name=name,
            description=description,
            email_address=email_address,
            connector_type=connector_type,
            team_members=team_members or [],
            admins=admins or [],
            settings=settings or {},
            created_at=now,
            updated_at=now,
            created_by=created_by,
        )

        # Persist to store if available
        store = _get_store()
        if store:
            try:
                store.create_shared_inbox(
                    inbox_id=inbox_id,
                    workspace_id=workspace_id,
                    name=name,
                    description=description,
                    email_address=email_address,
                    connector_type=connector_type,
                    team_members=team_members or [],
                    admins=admins or [],
                    settings=settings or {},
                    created_by=created_by,
                )
            except Exception as e:
                logger.warning(f"[SharedInbox] Failed to persist inbox to store: {e}")

        # Always keep in-memory cache for fast reads
        with _storage_lock:
            _shared_inboxes[inbox_id] = inbox
            _inbox_messages[inbox_id] = {}

        logger.info(f"[SharedInbox] Created inbox {inbox_id}: {name}")

        return {
            "success": True,
            "inbox": inbox.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Failed to create shared inbox: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_list_shared_inboxes(
    workspace_id: str,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List shared inboxes the user has access to.

    GET /api/v1/inbox/shared?workspace_id=ws_123
    """
    try:
        # Try persistent store first
        store = _get_store()
        if store:
            try:
                stored_inboxes = store.list_shared_inboxes(workspace_id, user_id)
                if stored_inboxes:
                    # Update in-memory cache
                    for inbox_data in stored_inboxes:
                        inbox_id = inbox_data.get("id")
                        if inbox_id and inbox_id not in _shared_inboxes:
                            # Reconstruct SharedInbox object for cache
                            with _storage_lock:
                                if inbox_id not in _inbox_messages:
                                    _inbox_messages[inbox_id] = {}
                    return {
                        "success": True,
                        "inboxes": stored_inboxes,
                        "total": len(stored_inboxes),
                    }
            except Exception as e:
                logger.warning(f"[SharedInbox] Failed to load from store, using cache: {e}")

        # Fallback to in-memory
        with _storage_lock:
            inboxes = [
                inbox.to_dict()
                for inbox in _shared_inboxes.values()
                if inbox.workspace_id == workspace_id
                and (user_id is None or user_id in inbox.team_members or user_id in inbox.admins)
            ]

        return {
            "success": True,
            "inboxes": inboxes,
            "total": len(inboxes),
        }

    except Exception as e:
        logger.exception(f"Failed to list shared inboxes: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_get_shared_inbox(
    inbox_id: str,
) -> Dict[str, Any]:
    """
    Get shared inbox details.

    GET /api/v1/inbox/shared/:id
    """
    try:
        # Try persistent store first
        store = _get_store()
        if store:
            try:
                inbox_data = store.get_shared_inbox(inbox_id)
                if inbox_data:
                    return {
                        "success": True,
                        "inbox": inbox_data,
                    }
            except Exception as e:
                logger.warning(f"[SharedInbox] Failed to load from store: {e}")

        # Fallback to in-memory
        with _storage_lock:
            inbox = _shared_inboxes.get(inbox_id)
            if not inbox:
                return {"success": False, "error": "Inbox not found"}

            # Update counts
            messages = _inbox_messages.get(inbox_id, {})
            inbox.message_count = len(messages)
            inbox.unread_count = sum(1 for m in messages.values() if m.status == MessageStatus.OPEN)

            return {
                "success": True,
                "inbox": inbox.to_dict(),
            }

    except Exception as e:
        logger.exception(f"Failed to get shared inbox: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_get_inbox_messages(
    inbox_id: str,
    status: Optional[str] = None,
    assigned_to: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Get messages in a shared inbox.

    GET /api/v1/inbox/shared/:id/messages
    Query params: status, assigned_to, tag, limit, offset
    """
    try:
        # Try persistent store first
        store = _get_store()
        if store:
            try:
                messages_data = store.get_inbox_messages(
                    inbox_id=inbox_id,
                    status=status,
                    assigned_to=assigned_to,
                    limit=limit,
                    offset=offset,
                )
                if messages_data is not None:
                    # Apply tag filter (not in store query)
                    if tag:
                        messages_data = [m for m in messages_data if tag in m.get("tags", [])]
                    return {
                        "success": True,
                        "messages": messages_data,
                        "total": len(messages_data),  # TODO: get actual total from store
                        "limit": limit,
                        "offset": offset,
                    }
            except Exception as e:
                logger.warning(f"[SharedInbox] Failed to load messages from store: {e}")

        # Fallback to in-memory
        with _storage_lock:
            if inbox_id not in _inbox_messages:
                return {"success": False, "error": "Inbox not found"}

            messages = list(_inbox_messages[inbox_id].values())

        # Filter
        if status:
            messages = [m for m in messages if m.status.value == status]
        if assigned_to:
            messages = [m for m in messages if m.assigned_to == assigned_to]
        if tag:
            messages = [m for m in messages if tag in m.tags]

        # Sort by received_at descending
        messages.sort(key=lambda m: m.received_at, reverse=True)

        # Paginate
        total = len(messages)
        messages = messages[offset : offset + limit]

        return {
            "success": True,
            "messages": [m.to_dict() for m in messages],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    except Exception as e:
        logger.exception(f"Failed to get inbox messages: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_assign_message(
    inbox_id: str,
    message_id: str,
    assigned_to: str,
    assigned_by: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Assign a message to a team member.

    POST /api/v1/inbox/shared/:id/messages/:msg_id/assign
    {
        "assigned_to": "user_123"
    }
    """
    try:
        now = datetime.now(timezone.utc)
        new_status = None

        with _storage_lock:
            messages = _inbox_messages.get(inbox_id, {})
            message = messages.get(message_id)

            if not message:
                return {"success": False, "error": "Message not found"}

            message.assigned_to = assigned_to
            message.assigned_at = now
            if message.status == MessageStatus.OPEN:
                message.status = MessageStatus.ASSIGNED
                new_status = MessageStatus.ASSIGNED.value

        # Persist to store if available
        store = _get_store()
        if store:
            try:
                updates = {
                    "assigned_to": assigned_to,
                    "assigned_at": now.isoformat(),
                }
                if new_status:
                    updates["status"] = new_status
                store.update_message(message_id, updates)
            except Exception as e:
                logger.warning(f"[SharedInbox] Failed to persist assignment to store: {e}")

        logger.info(f"[SharedInbox] Assigned message {message_id} to {assigned_to}")

        return {
            "success": True,
            "message": message.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Failed to assign message: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_update_message_status(
    inbox_id: str,
    message_id: str,
    status: str,
    updated_by: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update message status.

    POST /api/v1/inbox/shared/:id/messages/:msg_id/status
    {
        "status": "resolved"
    }
    """
    try:
        now = datetime.now(timezone.utc)
        is_resolved = False

        with _storage_lock:
            messages = _inbox_messages.get(inbox_id, {})
            message = messages.get(message_id)

            if not message:
                return {"success": False, "error": "Message not found"}

            message.status = MessageStatus(status)

            if message.status == MessageStatus.RESOLVED:
                message.resolved_at = now
                message.resolved_by = updated_by
                is_resolved = True

        # Persist to store if available
        store = _get_store()
        if store:
            try:
                updates = {"status": status}
                if is_resolved:
                    updates["resolved_at"] = now.isoformat()
                    updates["resolved_by"] = updated_by
                store.update_message(message_id, updates)
            except Exception as e:
                logger.warning(f"[SharedInbox] Failed to persist status to store: {e}")

        logger.info(f"[SharedInbox] Updated message {message_id} status to {status}")

        return {
            "success": True,
            "message": message.to_dict(),
        }

    except ValueError:
        return {"success": False, "error": f"Invalid status: {status}"}
    except Exception as e:
        logger.exception(f"Failed to update message status: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_add_message_tag(
    inbox_id: str,
    message_id: str,
    tag: str,
) -> Dict[str, Any]:
    """
    Add a tag to a message.

    POST /api/v1/inbox/shared/:id/messages/:msg_id/tag
    {
        "tag": "urgent"
    }
    """
    try:
        with _storage_lock:
            messages = _inbox_messages.get(inbox_id, {})
            message = messages.get(message_id)

            if not message:
                return {"success": False, "error": "Message not found"}

            if tag not in message.tags:
                message.tags.append(tag)

        return {
            "success": True,
            "message": message.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Failed to add message tag: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_add_message_to_inbox(
    inbox_id: str,
    email_id: str,
    subject: str,
    from_address: str,
    to_addresses: List[str],
    snippet: str,
    received_at: Optional[datetime] = None,
    thread_id: Optional[str] = None,
    priority: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Add a message to a shared inbox (used by sync/routing).

    Internal API - not directly exposed to users.
    """
    try:
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        actual_received_at = received_at or datetime.now(timezone.utc)

        message = SharedInboxMessage(
            id=message_id,
            inbox_id=inbox_id,
            email_id=email_id,
            subject=subject,
            from_address=from_address,
            to_addresses=to_addresses,
            snippet=snippet,
            received_at=actual_received_at,
            thread_id=thread_id,
            priority=priority,
        )

        # Persist to store if available
        store = _get_store()
        if store:
            try:
                # Get workspace_id from inbox if not provided
                if not workspace_id:
                    inbox_data = store.get_shared_inbox(inbox_id)
                    workspace_id = (
                        inbox_data.get("workspace_id", "default") if inbox_data else "default"
                    )

                store.save_message(
                    message_id=message_id,
                    inbox_id=inbox_id,
                    workspace_id=workspace_id,
                    email_id=email_id,
                    subject=subject,
                    from_address=from_address,
                    to_addresses=to_addresses,
                    snippet=snippet,
                    received_at=actual_received_at,
                    thread_id=thread_id,
                    priority=priority,
                )
            except Exception as e:
                logger.warning(f"[SharedInbox] Failed to persist message to store: {e}")

        # Keep in-memory cache
        with _storage_lock:
            if inbox_id not in _inbox_messages:
                _inbox_messages[inbox_id] = {}

            _inbox_messages[inbox_id][message_id] = message

        return {
            "success": True,
            "message": message.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Failed to add message to inbox: {e}")
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# Routing Rule Handlers
# =============================================================================


async def handle_create_routing_rule(
    workspace_id: str,
    name: str,
    conditions: List[Dict[str, Any]],
    actions: List[Dict[str, Any]],
    condition_logic: str = "AND",
    priority: int = 5,
    enabled: bool = True,
    description: Optional[str] = None,
    created_by: Optional[str] = None,
    inbox_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a routing rule.

    POST /api/v1/inbox/routing/rules
    {
        "workspace_id": "ws_123",
        "name": "Urgent Customer Issues",
        "conditions": [
            {"field": "subject", "operator": "contains", "value": "urgent"}
        ],
        "condition_logic": "AND",
        "actions": [
            {"type": "assign", "target": "support-team"},
            {"type": "label", "target": "urgent"}
        ],
        "priority": 1
    }
    """
    try:
        rule_id = f"rule_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        # Prepare rule data for persistent storage
        rule_data = {
            "id": rule_id,
            "name": name,
            "workspace_id": workspace_id,
            "inbox_id": inbox_id,
            "conditions": conditions,
            "condition_logic": condition_logic,
            "actions": actions,
            "priority": priority,
            "enabled": enabled,
            "description": description,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "created_by": created_by,
            "stats": {"total_matches": 0, "matched": 0, "applied": 0},
        }

        # Use RulesStore for persistent storage (primary)
        rules_store = _get_rules_store()
        if rules_store:
            try:
                rules_store.create_rule(rule_data)
                logger.info(f"[SharedInbox] Created routing rule {rule_id}: {name} (persistent)")
            except Exception as e:
                logger.warning(f"[SharedInbox] Failed to persist rule to RulesStore: {e}")
                # Fall through to in-memory storage

        # Also persist to email store for backward compatibility
        email_store = _get_store()
        if email_store:
            try:
                email_store.create_routing_rule(
                    rule_id=rule_id,
                    workspace_id=workspace_id,
                    name=name,
                    conditions=conditions,
                    actions=actions,
                    priority=priority,
                    enabled=enabled,
                    description=description,
                    inbox_id=inbox_id,
                )
            except Exception as e:
                logger.warning(f"[SharedInbox] Failed to persist rule to email store: {e}")

        # Build in-memory RoutingRule object for cache
        rule = RoutingRule(
            id=rule_id,
            workspace_id=workspace_id,
            name=name,
            conditions=[RuleCondition.from_dict(c) for c in conditions],
            condition_logic=condition_logic,
            actions=[RuleAction.from_dict(a) for a in actions],
            priority=priority,
            enabled=enabled,
            description=description,
            created_at=now,
            updated_at=now,
            created_by=created_by,
            stats={"total_matches": 0},
        )

        # Keep in-memory cache for fast reads
        with _storage_lock:
            _routing_rules[rule_id] = rule

        return {
            "success": True,
            "rule": rule.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Failed to create routing rule: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_list_routing_rules(
    workspace_id: str,
    enabled_only: bool = False,
    limit: int = 100,
    offset: int = 0,
    inbox_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List routing rules for a workspace.

    GET /api/v1/inbox/routing/rules?workspace_id=ws_123
    """
    try:
        # Try RulesStore first (primary persistent storage)
        rules_store = _get_rules_store()
        if rules_store:
            try:
                rules = rules_store.list_rules(
                    workspace_id=workspace_id,
                    inbox_id=inbox_id,
                    enabled_only=enabled_only,
                    limit=limit,
                    offset=offset,
                )
                total = rules_store.count_rules(
                    workspace_id=workspace_id,
                    inbox_id=inbox_id,
                    enabled_only=enabled_only,
                )
                return {
                    "success": True,
                    "rules": rules,
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }
            except Exception as e:
                logger.warning(f"[SharedInbox] Failed to load rules from RulesStore: {e}")

        # Try email store as fallback
        email_store = _get_store()
        if email_store:
            try:
                rules = email_store.list_routing_rules(
                    workspace_id=workspace_id,
                    inbox_id=inbox_id,
                    enabled_only=enabled_only,
                )
                if rules is not None:
                    total = len(rules)
                    rules = rules[offset : offset + limit]
                    return {
                        "success": True,
                        "rules": rules,
                        "total": total,
                        "limit": limit,
                        "offset": offset,
                    }
            except Exception as e:
                logger.warning(f"[SharedInbox] Failed to load rules from email store: {e}")

        # Fallback to in-memory
        with _storage_lock:
            all_rules = [
                rule.to_dict()
                for rule in sorted(_routing_rules.values(), key=lambda r: r.priority)
                if rule.workspace_id == workspace_id
            ]
            if inbox_id:
                all_rules = [
                    r
                    for r in all_rules
                    if r.get("inbox_id") == inbox_id or r.get("inbox_id") is None
                ]
            if enabled_only:
                all_rules = [r for r in all_rules if r.get("enabled", True)]
            total = len(all_rules)
            rules = all_rules[offset : offset + limit]

        return {
            "success": True,
            "rules": rules,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    except Exception as e:
        logger.exception(f"Failed to list routing rules: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_update_routing_rule(
    rule_id: str,
    updates: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Update a routing rule.

    PATCH /api/v1/inbox/routing/rules/:id
    {
        "enabled": false,
        "priority": 2
    }
    """
    try:
        updated_rule_data = None

        # Update in RulesStore first (primary persistent storage)
        rules_store = _get_rules_store()
        if rules_store:
            try:
                updated_rule_data = rules_store.update_rule(rule_id, updates)
                if updated_rule_data:
                    logger.info(f"[SharedInbox] Updated routing rule {rule_id} in RulesStore")
            except Exception as e:
                logger.warning(f"[SharedInbox] Failed to update rule in RulesStore: {e}")

        # Also update in email store for backward compatibility
        email_store = _get_store()
        if email_store:
            try:
                email_store.update_routing_rule(rule_id, **updates)
            except Exception as e:
                logger.warning(f"[SharedInbox] Failed to update rule in email store: {e}")

        # Update in-memory cache
        with _storage_lock:
            rule = _routing_rules.get(rule_id)
            if rule:
                # Update fields
                if "name" in updates:
                    rule.name = updates["name"]
                if "description" in updates:
                    rule.description = updates["description"]
                if "conditions" in updates:
                    rule.conditions = [RuleCondition.from_dict(c) for c in updates["conditions"]]
                if "condition_logic" in updates:
                    rule.condition_logic = updates["condition_logic"]
                if "actions" in updates:
                    rule.actions = [RuleAction.from_dict(a) for a in updates["actions"]]
                if "priority" in updates:
                    rule.priority = updates["priority"]
                if "enabled" in updates:
                    rule.enabled = updates["enabled"]

                rule.updated_at = datetime.now(timezone.utc)

        # Return data from persistent storage if available
        if updated_rule_data:
            return {
                "success": True,
                "rule": updated_rule_data,
            }

        # Fallback: try to get from RulesStore
        if rules_store:
            try:
                rule_data = rules_store.get_rule(rule_id)
                if rule_data:
                    return {
                        "success": True,
                        "rule": rule_data,
                    }
            except Exception:
                pass

        # Return from in-memory cache if available
        with _storage_lock:
            rule = _routing_rules.get(rule_id)
            if rule:
                return {
                    "success": True,
                    "rule": rule.to_dict(),
                }

        return {"success": False, "error": "Rule not found"}

    except Exception as e:
        logger.exception(f"Failed to update routing rule: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_delete_routing_rule(
    rule_id: str,
) -> Dict[str, Any]:
    """
    Delete a routing rule.

    DELETE /api/v1/inbox/routing/rules/:id
    """
    try:
        deleted = False

        # Delete from RulesStore (primary persistent storage)
        rules_store = _get_rules_store()
        if rules_store:
            try:
                deleted = rules_store.delete_rule(rule_id)
                if deleted:
                    logger.info(f"[SharedInbox] Deleted routing rule {rule_id} from RulesStore")
            except Exception as e:
                logger.warning(f"[SharedInbox] Failed to delete rule from RulesStore: {e}")

        # Also delete from email store for backward compatibility
        email_store = _get_store()
        if email_store:
            try:
                email_store.delete_routing_rule(rule_id)
            except Exception as e:
                logger.warning(f"[SharedInbox] Failed to delete rule from email store: {e}")

        # Delete from in-memory cache
        with _storage_lock:
            if rule_id in _routing_rules:
                del _routing_rules[rule_id]
                deleted = True

        if deleted:
            return {
                "success": True,
                "deleted": rule_id,
            }
        else:
            return {
                "success": False,
                "error": "Rule not found",
            }

    except Exception as e:
        logger.exception(f"Failed to delete routing rule: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_test_routing_rule(
    rule_id: str,
    workspace_id: str,
) -> Dict[str, Any]:
    """
    Test a routing rule against existing messages.

    POST /api/v1/inbox/routing/rules/:id/test
    {
        "workspace_id": "ws_123"
    }
    """
    try:
        rule = None
        rule_data = None

        # Try RulesStore first (primary persistent storage)
        rules_store = _get_rules_store()
        if rules_store:
            try:
                rule_data = rules_store.get_rule(rule_id)
                if rule_data:
                    rule = RoutingRule.from_dict(rule_data)
            except Exception as e:
                logger.warning(f"[SharedInbox] Failed to load rule from RulesStore: {e}")

        # Try email store as fallback
        if not rule:
            email_store = _get_store()
            if email_store:
                try:
                    rule_data = email_store.get_routing_rule(rule_id)
                    if rule_data:
                        rule = RoutingRule.from_dict(rule_data)
                except Exception as e:
                    logger.warning(f"[SharedInbox] Failed to load rule from email store: {e}")

        # Fallback to in-memory
        if not rule:
            with _storage_lock:
                rule = _routing_rules.get(rule_id)
                if not rule:
                    return {"success": False, "error": "Rule not found"}

        # Count matching messages across all inboxes in workspace
        match_count = 0
        with _storage_lock:
            for inbox_id, messages in _inbox_messages.items():
                inbox = _shared_inboxes.get(inbox_id)
                if inbox and inbox.workspace_id == workspace_id:
                    for message in messages.values():
                        if _evaluate_rule(rule, message):
                            match_count += 1

        return {
            "success": True,
            "rule_id": rule_id,
            "match_count": match_count,
            "rule": rule.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Failed to test routing rule: {e}")
        return {
            "success": False,
            "error": str(e),
        }


def _evaluate_rule(rule: RoutingRule, message: SharedInboxMessage) -> bool:
    """Evaluate if a routing rule matches a message."""
    import re

    results = []

    for condition in rule.conditions:
        value = ""
        if condition.field == RuleConditionField.FROM:
            value = message.from_address.lower()
        elif condition.field == RuleConditionField.TO:
            value = " ".join(message.to_addresses).lower()
        elif condition.field == RuleConditionField.SUBJECT:
            value = message.subject.lower()
        elif condition.field == RuleConditionField.SENDER_DOMAIN:
            value = (
                message.from_address.split("@")[-1].lower() if "@" in message.from_address else ""
            )
        elif condition.field == RuleConditionField.PRIORITY:
            value = message.priority or ""

        condition_value = condition.value.lower()
        matched = False

        if condition.operator == RuleConditionOperator.CONTAINS:
            matched = condition_value in value
        elif condition.operator == RuleConditionOperator.EQUALS:
            matched = value == condition_value
        elif condition.operator == RuleConditionOperator.STARTS_WITH:
            matched = value.startswith(condition_value)
        elif condition.operator == RuleConditionOperator.ENDS_WITH:
            matched = value.endswith(condition_value)
        elif condition.operator == RuleConditionOperator.MATCHES:
            try:
                matched = bool(re.search(condition_value, value, re.IGNORECASE))
            except re.error:
                matched = False

        results.append(matched)

    if rule.condition_logic == "AND":
        return all(results) if results else False
    else:  # OR
        return any(results) if results else False


async def get_matching_rules_for_email(
    inbox_id: str,
    email_data: Dict[str, Any],
    workspace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get all matching routing rules for an email message.

    This function queries the RulesStore to find enabled rules that match
    the given email data, sorted by priority.

    Args:
        inbox_id: The inbox ID to get rules for
        email_data: Email data dictionary with keys:
            - from_address: Sender email address
            - to_addresses: List of recipient addresses
            - subject: Email subject line
            - snippet: Email body preview
            - priority: Email priority level (optional)
        workspace_id: Optional workspace filter

    Returns:
        List of matching rule dictionaries sorted by priority (ascending)
    """
    matching_rules = []

    # Try RulesStore first (primary persistent storage)
    rules_store = _get_rules_store()
    if rules_store:
        try:
            matching_rules = rules_store.get_matching_rules(
                inbox_id=inbox_id,
                email_data=email_data,
                workspace_id=workspace_id,
            )
            return matching_rules
        except Exception as e:
            logger.warning(f"[SharedInbox] Failed to get matching rules from RulesStore: {e}")

    # Fallback to in-memory evaluation
    with _storage_lock:
        # Get all enabled rules for this workspace/inbox
        rules = [
            rule
            for rule in _routing_rules.values()
            if rule.enabled and (workspace_id is None or rule.workspace_id == workspace_id)
        ]
        # Sort by priority
        rules.sort(key=lambda r: r.priority)

        # Create a message-like object for evaluation
        class _EmailMessage:
            def __init__(self, data: Dict[str, Any]):
                self.from_address = data.get("from_address", "")
                self.to_addresses = data.get("to_addresses", [])
                self.subject = data.get("subject", "")
                self.snippet = data.get("snippet", "")
                self.priority = data.get("priority")

        msg = _EmailMessage(email_data)

        for rule in rules:
            if _evaluate_rule(rule, msg):
                matching_rules.append(rule.to_dict())

    return matching_rules


async def apply_routing_rules_to_message(
    inbox_id: str,
    message: SharedInboxMessage,
    workspace_id: str,
) -> Dict[str, Any]:
    """
    Apply matching routing rules to a message.

    Args:
        inbox_id: The inbox containing the message
        message: The message to apply rules to
        workspace_id: The workspace ID

    Returns:
        Dictionary with applied actions and any changes made
    """
    # Build email data from message
    email_data = {
        "from_address": message.from_address,
        "to_addresses": message.to_addresses,
        "subject": message.subject,
        "snippet": message.snippet,
        "priority": message.priority,
    }

    # Get matching rules
    matching_rules = await get_matching_rules_for_email(
        inbox_id=inbox_id,
        email_data=email_data,
        workspace_id=workspace_id,
    )

    if not matching_rules:
        return {"applied": False, "rules_matched": 0, "actions": []}

    applied_actions = []
    changes_made = {}

    for rule in matching_rules:
        actions = rule.get("actions", [])
        for action in actions:
            action_type = action.get("type")
            target = action.get("target")
            _ = action.get("params", {})  # Reserved for future action params

            if action_type == "assign" and target:
                message.assigned_to = target
                message.assigned_at = datetime.now(timezone.utc)
                if message.status == MessageStatus.OPEN:
                    message.status = MessageStatus.ASSIGNED
                changes_made["assigned_to"] = target
                applied_actions.append({"type": "assign", "target": target})

            elif action_type == "label" and target:
                if target not in message.tags:
                    message.tags.append(target)
                applied_actions.append({"type": "label", "target": target})

            elif action_type == "escalate":
                message.priority = "high"
                changes_made["priority"] = "high"
                applied_actions.append({"type": "escalate"})

            elif action_type == "archive":
                message.status = MessageStatus.CLOSED
                changes_made["status"] = "closed"
                applied_actions.append({"type": "archive"})

        # Update rule stats
        rules_store = _get_rules_store()
        if rules_store:
            try:
                rules_store.increment_rule_stats(rule["id"], matched=0, applied=1)
            except Exception:
                pass

    return {
        "applied": bool(applied_actions),
        "rules_matched": len(matching_rules),
        "actions": applied_actions,
        "changes": changes_made,
    }


# =============================================================================
# Handler Class
# =============================================================================


class SharedInboxHandler(BaseHandler):
    """
    HTTP handler for shared inbox endpoints.

    Integrates with the Aragora server routing system.
    """

    ROUTES = [
        "/api/v1/inbox/shared",
        "/api/v1/inbox/routing/rules",
    ]

    ROUTE_PREFIXES = [
        "/api/v1/inbox/shared/",
        "/api/v1/inbox/routing/rules/",
    ]

    def __init__(self, ctx: Dict[str, Any]):
        """Initialize with server context."""
        super().__init__(ctx)  # type: ignore[arg-type]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        if path in self.ROUTES:
            return True
        for prefix in self.ROUTE_PREFIXES:
            if path.startswith(prefix):
                return True
        return False

    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route shared inbox endpoint requests."""
        return None

    async def handle_post_shared_inbox(self, data: Dict[str, Any]) -> HandlerResult:
        """POST /api/v1/inbox/shared"""
        workspace_id = data.get("workspace_id")
        name = data.get("name")

        if not workspace_id or not name:
            return error_response("workspace_id and name required", 400)

        result = await handle_create_shared_inbox(
            workspace_id=workspace_id,
            name=name,
            description=data.get("description"),
            email_address=data.get("email_address"),
            connector_type=data.get("connector_type"),
            team_members=data.get("team_members"),
            admins=data.get("admins"),
            settings=data.get("settings"),
            created_by=self._get_user_id(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_get_shared_inboxes(self, params: Dict[str, Any]) -> HandlerResult:
        """GET /api/v1/inbox/shared"""
        workspace_id = params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id required", 400)

        result = await handle_list_shared_inboxes(
            workspace_id=workspace_id,
            user_id=self._get_user_id(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_get_shared_inbox(self, params: Dict[str, Any], inbox_id: str) -> HandlerResult:
        """GET /api/v1/inbox/shared/:id"""
        result = await handle_get_shared_inbox(inbox_id)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 404)

    async def handle_get_inbox_messages(
        self, params: Dict[str, Any], inbox_id: str
    ) -> HandlerResult:
        """GET /api/v1/inbox/shared/:id/messages"""
        result = await handle_get_inbox_messages(
            inbox_id=inbox_id,
            status=params.get("status"),
            assigned_to=params.get("assigned_to"),
            tag=params.get("tag"),
            limit=int(params.get("limit", 50)),
            offset=int(params.get("offset", 0)),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_assign_message(
        self, data: Dict[str, Any], inbox_id: str, message_id: str
    ) -> HandlerResult:
        """POST /api/v1/inbox/shared/:id/messages/:msg_id/assign"""
        assigned_to = data.get("assigned_to")
        if not assigned_to:
            return error_response("assigned_to required", 400)

        result = await handle_assign_message(
            inbox_id=inbox_id,
            message_id=message_id,
            assigned_to=assigned_to,
            assigned_by=self._get_user_id(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_update_status(
        self, data: Dict[str, Any], inbox_id: str, message_id: str
    ) -> HandlerResult:
        """POST /api/v1/inbox/shared/:id/messages/:msg_id/status"""
        status = data.get("status")
        if not status:
            return error_response("status required", 400)

        result = await handle_update_message_status(
            inbox_id=inbox_id,
            message_id=message_id,
            status=status,
            updated_by=self._get_user_id(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_add_tag(
        self, data: Dict[str, Any], inbox_id: str, message_id: str
    ) -> HandlerResult:
        """POST /api/v1/inbox/shared/:id/messages/:msg_id/tag"""
        tag = data.get("tag")
        if not tag:
            return error_response("tag required", 400)

        result = await handle_add_message_tag(
            inbox_id=inbox_id,
            message_id=message_id,
            tag=tag,
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_routing_rule(self, data: Dict[str, Any]) -> HandlerResult:
        """POST /api/v1/inbox/routing/rules"""
        workspace_id = data.get("workspace_id")
        name = data.get("name")
        conditions = data.get("conditions", [])
        actions = data.get("actions", [])

        if not workspace_id or not name or not conditions or not actions:
            return error_response("workspace_id, name, conditions, and actions required", 400)

        result = await handle_create_routing_rule(
            workspace_id=workspace_id,
            name=name,
            conditions=conditions,
            actions=actions,
            condition_logic=data.get("condition_logic", "AND"),
            priority=data.get("priority", 5),
            enabled=data.get("enabled", True),
            description=data.get("description"),
            created_by=self._get_user_id(),
            inbox_id=data.get("inbox_id"),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_get_routing_rules(self, params: Dict[str, Any]) -> HandlerResult:
        """GET /api/v1/inbox/routing/rules"""
        workspace_id = params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id required", 400)

        result = await handle_list_routing_rules(
            workspace_id=workspace_id,
            enabled_only=params.get("enabled_only", "false").lower() == "true",
            limit=int(params.get("limit", 100)),
            offset=int(params.get("offset", 0)),
            inbox_id=params.get("inbox_id"),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_patch_routing_rule(self, data: Dict[str, Any], rule_id: str) -> HandlerResult:
        """PATCH /api/v1/inbox/routing/rules/:id"""
        result = await handle_update_routing_rule(rule_id, data)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_delete_routing_rule(self, rule_id: str) -> HandlerResult:
        """DELETE /api/v1/inbox/routing/rules/:id"""
        result = await handle_delete_routing_rule(rule_id)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_test_routing_rule(
        self, data: Dict[str, Any], rule_id: str
    ) -> HandlerResult:
        """POST /api/v1/inbox/routing/rules/:id/test"""
        workspace_id = data.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id required", 400)

        result = await handle_test_routing_rule(rule_id, workspace_id)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    def _get_user_id(self) -> str:
        """Get user ID from auth context."""
        auth_ctx = self.ctx.get("auth_context")
        if auth_ctx and hasattr(auth_ctx, "user_id"):
            return auth_ctx.user_id
        return "default"
