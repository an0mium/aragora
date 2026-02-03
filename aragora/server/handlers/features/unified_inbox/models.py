"""Unified Inbox data models.

Enums and dataclasses for the unified inbox feature:
- EmailProvider, AccountStatus, TriageAction enums
- ConnectedAccount, UnifiedMessage, TriageResult, InboxStats dataclasses
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# =============================================================================
# Enums
# =============================================================================


class EmailProvider(Enum):
    """Supported email providers."""

    GMAIL = "gmail"
    OUTLOOK = "outlook"


class AccountStatus(Enum):
    """Account connection status."""

    PENDING = "pending"
    CONNECTED = "connected"
    SYNCING = "syncing"
    ERROR = "error"
    DISCONNECTED = "disconnected"


class TriageAction(Enum):
    """Available triage actions."""

    RESPOND_URGENT = "respond_urgent"
    RESPOND_NORMAL = "respond_normal"
    DELEGATE = "delegate"
    SCHEDULE = "schedule"
    ARCHIVE = "archive"
    DELETE = "delete"
    FLAG = "flag"
    DEFER = "defer"


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class ConnectedAccount:
    """Represents a connected email account."""

    id: str
    provider: EmailProvider
    email_address: str
    display_name: str
    status: AccountStatus
    connected_at: datetime
    last_sync: datetime | None = None
    total_messages: int = 0
    unread_count: int = 0
    sync_errors: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "provider": self.provider.value,
            "email_address": self.email_address,
            "display_name": self.display_name,
            "status": self.status.value,
            "connected_at": self.connected_at.isoformat(),
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "total_messages": self.total_messages,
            "unread_count": self.unread_count,
            "sync_errors": self.sync_errors,
        }


@dataclass
class UnifiedMessage:
    """Unified message representation across providers."""

    id: str
    account_id: str
    provider: EmailProvider
    external_id: str  # Provider-specific ID
    subject: str
    sender_email: str
    sender_name: str
    recipients: list[str]
    cc: list[str]
    received_at: datetime
    snippet: str
    body_preview: str
    is_read: bool
    is_starred: bool
    has_attachments: bool
    labels: list[str]
    thread_id: str | None = None
    # Priority scoring
    priority_score: float = 0.5
    priority_tier: str = "medium"
    priority_reasons: list[str] = field(default_factory=list)
    # Triage results
    triage_action: TriageAction | None = None
    triage_rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "account_id": self.account_id,
            "provider": self.provider.value,
            "external_id": self.external_id,
            "subject": self.subject,
            "sender": {
                "email": self.sender_email,
                "name": self.sender_name,
            },
            "recipients": self.recipients,
            "cc": self.cc,
            "received_at": self.received_at.isoformat(),
            "snippet": self.snippet,
            "is_read": self.is_read,
            "is_starred": self.is_starred,
            "has_attachments": self.has_attachments,
            "labels": self.labels,
            "thread_id": self.thread_id,
            "priority": {
                "score": self.priority_score,
                "tier": self.priority_tier,
                "reasons": self.priority_reasons,
            },
            "triage": (
                {
                    "action": self.triage_action.value if self.triage_action else None,
                    "rationale": self.triage_rationale,
                }
                if self.triage_action
                else None
            ),
        }


@dataclass
class TriageResult:
    """Result of multi-agent triage."""

    message_id: str
    recommended_action: TriageAction
    confidence: float
    rationale: str
    suggested_response: str | None
    delegate_to: str | None
    schedule_for: datetime | None
    agents_involved: list[str]
    debate_summary: str | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "message_id": self.message_id,
            "recommended_action": self.recommended_action.value,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "suggested_response": self.suggested_response,
            "delegate_to": self.delegate_to,
            "schedule_for": self.schedule_for.isoformat() if self.schedule_for else None,
            "agents_involved": self.agents_involved,
            "debate_summary": self.debate_summary,
        }


@dataclass
class InboxStats:
    """Inbox health statistics."""

    total_accounts: int
    total_messages: int
    unread_count: int
    messages_by_priority: dict[str, int]
    messages_by_provider: dict[str, int]
    avg_response_time_hours: float
    pending_triage: int
    sync_health: dict[str, Any]
    top_senders: list[dict[str, Any]]
    hourly_volume: list[dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_accounts": self.total_accounts,
            "total_messages": self.total_messages,
            "unread_count": self.unread_count,
            "messages_by_priority": self.messages_by_priority,
            "messages_by_provider": self.messages_by_provider,
            "avg_response_time_hours": self.avg_response_time_hours,
            "pending_triage": self.pending_triage,
            "sync_health": self.sync_health,
            "top_senders": self.top_senders,
            "hourly_volume": self.hourly_volume,
        }


# =============================================================================
# Record Conversion Helpers
# =============================================================================


def ensure_datetime(value: Any) -> datetime | None:
    """Convert a value to datetime, returning None if not possible."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def account_to_record(account: ConnectedAccount) -> dict[str, Any]:
    """Serialize a ConnectedAccount to a store record."""
    return {
        "id": account.id,
        "provider": account.provider.value,
        "email_address": account.email_address,
        "display_name": account.display_name,
        "status": account.status.value,
        "connected_at": account.connected_at,
        "last_sync": account.last_sync,
        "total_messages": account.total_messages,
        "unread_count": account.unread_count,
        "sync_errors": account.sync_errors,
        "metadata": account.metadata,
    }


def record_to_account(record: dict[str, Any]) -> ConnectedAccount:
    """Deserialize a store record to a ConnectedAccount."""
    return ConnectedAccount(
        id=record["id"],
        provider=EmailProvider(record["provider"]),
        email_address=record.get("email_address", ""),
        display_name=record.get("display_name", ""),
        status=AccountStatus(record.get("status", "pending")),
        connected_at=ensure_datetime(record.get("connected_at")) or datetime.now(timezone.utc),
        last_sync=ensure_datetime(record.get("last_sync")),
        total_messages=int(record.get("total_messages", 0)),
        unread_count=int(record.get("unread_count", 0)),
        sync_errors=int(record.get("sync_errors", 0)),
        metadata=record.get("metadata") or {},
    )


def message_to_record(message: UnifiedMessage) -> dict[str, Any]:
    """Serialize a UnifiedMessage to a store record."""
    return {
        "id": message.id,
        "account_id": message.account_id,
        "provider": message.provider.value,
        "external_id": message.external_id,
        "subject": message.subject,
        "sender_email": message.sender_email,
        "sender_name": message.sender_name,
        "recipients": message.recipients,
        "cc": message.cc,
        "received_at": message.received_at,
        "snippet": message.snippet,
        "body_preview": message.body_preview,
        "is_read": message.is_read,
        "is_starred": message.is_starred,
        "has_attachments": message.has_attachments,
        "labels": message.labels,
        "thread_id": message.thread_id,
        "priority_score": message.priority_score,
        "priority_tier": message.priority_tier,
        "priority_reasons": message.priority_reasons,
        "triage_action": message.triage_action.value if message.triage_action else None,
        "triage_rationale": message.triage_rationale,
    }


def record_to_message(record: dict[str, Any]) -> UnifiedMessage:
    """Deserialize a store record to a UnifiedMessage."""
    triage_action = record.get("triage_action")
    return UnifiedMessage(
        id=record["id"],
        account_id=record["account_id"],
        provider=EmailProvider(record["provider"]),
        external_id=record.get("external_id", ""),
        subject=record.get("subject", ""),
        sender_email=record.get("sender_email", ""),
        sender_name=record.get("sender_name", ""),
        recipients=record.get("recipients") or [],
        cc=record.get("cc") or [],
        received_at=ensure_datetime(record.get("received_at")) or datetime.now(timezone.utc),
        snippet=record.get("snippet", ""),
        body_preview=record.get("body_preview", ""),
        is_read=bool(record.get("is_read")),
        is_starred=bool(record.get("is_starred")),
        has_attachments=bool(record.get("has_attachments")),
        labels=record.get("labels") or [],
        thread_id=record.get("thread_id"),
        priority_score=float(record.get("priority_score", 0.0)),
        priority_tier=record.get("priority_tier", "medium"),
        priority_reasons=record.get("priority_reasons") or [],
        triage_action=TriageAction(triage_action) if triage_action else None,
        triage_rationale=record.get("triage_rationale"),
    )


def triage_to_record(triage: TriageResult) -> dict[str, Any]:
    """Serialize a TriageResult to a store record."""
    return {
        "message_id": triage.message_id,
        "recommended_action": triage.recommended_action.value,
        "confidence": triage.confidence,
        "rationale": triage.rationale,
        "suggested_response": triage.suggested_response,
        "delegate_to": triage.delegate_to,
        "schedule_for": triage.schedule_for,
        "agents_involved": triage.agents_involved,
        "debate_summary": triage.debate_summary,
        "created_at": datetime.now(timezone.utc),
    }


def record_to_triage(record: dict[str, Any]) -> TriageResult:
    """Deserialize a store record to a TriageResult."""
    return TriageResult(
        message_id=record["message_id"],
        recommended_action=TriageAction(record["recommended_action"]),
        confidence=float(record.get("confidence", 0.0)),
        rationale=record.get("rationale", ""),
        suggested_response=record.get("suggested_response"),
        delegate_to=record.get("delegate_to"),
        schedule_for=ensure_datetime(record.get("schedule_for")),
        agents_involved=record.get("agents_involved") or [],
        debate_summary=record.get("debate_summary"),
    )
