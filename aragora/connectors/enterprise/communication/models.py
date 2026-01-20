"""
Email data models for communication connectors.

Provides dataclasses for emails, threads, attachments, and sync state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class EmailAttachment:
    """An email attachment."""

    id: str
    filename: str
    mime_type: str
    size: int
    data: Optional[bytes] = None  # Populated on demand


@dataclass
class GmailLabel:
    """A Gmail label/folder."""

    id: str
    name: str
    type: str = "user"  # "system" or "user"
    message_list_visibility: str = "show"
    label_list_visibility: str = "labelShow"


@dataclass
class EmailMessage:
    """
    An email message.

    Stores core email data plus AI-computed importance score.
    """

    id: str
    thread_id: str
    subject: str
    from_address: str
    to_addresses: List[str]
    date: datetime
    body_text: str
    body_html: str = ""
    snippet: str = ""
    labels: List[str] = field(default_factory=list)
    cc_addresses: List[str] = field(default_factory=list)
    bcc_addresses: List[str] = field(default_factory=list)
    attachments: List[EmailAttachment] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    is_read: bool = False
    is_starred: bool = False
    is_important: bool = False
    importance_score: float = 0.0  # AI-computed priority 0.0-1.0
    importance_reason: str = ""  # AI-generated explanation

    @property
    def message_id_header(self) -> Optional[str]:
        """Get the Message-ID header for reply threading."""
        return self.headers.get("Message-ID") or self.headers.get("Message-Id")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "subject": self.subject,
            "from_address": self.from_address,
            "to_addresses": self.to_addresses,
            "date": self.date.isoformat() if self.date else None,
            "body_text": self.body_text,
            "body_html": self.body_html,
            "snippet": self.snippet,
            "labels": self.labels,
            "cc_addresses": self.cc_addresses,
            "bcc_addresses": self.bcc_addresses,
            "attachments": [
                {"id": a.id, "filename": a.filename, "mime_type": a.mime_type, "size": a.size}
                for a in self.attachments
            ],
            "is_read": self.is_read,
            "is_starred": self.is_starred,
            "is_important": self.is_important,
            "importance_score": self.importance_score,
            "importance_reason": self.importance_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmailMessage":
        """Deserialize from dictionary."""
        date = data.get("date")
        if isinstance(date, str):
            date = datetime.fromisoformat(date)

        attachments = [
            EmailAttachment(
                id=a["id"],
                filename=a["filename"],
                mime_type=a["mime_type"],
                size=a["size"],
            )
            for a in data.get("attachments", [])
        ]

        return cls(
            id=data["id"],
            thread_id=data["thread_id"],
            subject=data.get("subject", ""),
            from_address=data.get("from_address", ""),
            to_addresses=data.get("to_addresses", []),
            date=date,
            body_text=data.get("body_text", ""),
            body_html=data.get("body_html", ""),
            snippet=data.get("snippet", ""),
            labels=data.get("labels", []),
            cc_addresses=data.get("cc_addresses", []),
            bcc_addresses=data.get("bcc_addresses", []),
            attachments=attachments,
            is_read=data.get("is_read", False),
            is_starred=data.get("is_starred", False),
            is_important=data.get("is_important", False),
            importance_score=data.get("importance_score", 0.0),
            importance_reason=data.get("importance_reason", ""),
        )


@dataclass
class EmailThread:
    """An email thread/conversation."""

    id: str
    subject: str
    messages: List[EmailMessage] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    last_message_date: Optional[datetime] = None
    snippet: str = ""
    message_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "subject": self.subject,
            "messages": [m.to_dict() for m in self.messages],
            "participants": self.participants,
            "labels": self.labels,
            "last_message_date": (
                self.last_message_date.isoformat() if self.last_message_date else None
            ),
            "snippet": self.snippet,
            "message_count": self.message_count,
        }


@dataclass
class GmailSyncState:
    """
    Gmail-specific sync state.

    Tracks history ID for incremental sync via Gmail History API.
    """

    user_id: str
    history_id: str = ""  # Gmail's incremental sync cursor
    last_sync: Optional[datetime] = None
    total_messages: int = 0
    indexed_messages: int = 0
    email_address: str = ""
    labels_synced: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "user_id": self.user_id,
            "history_id": self.history_id,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "total_messages": self.total_messages,
            "indexed_messages": self.indexed_messages,
            "email_address": self.email_address,
            "labels_synced": self.labels_synced,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GmailSyncState":
        """Deserialize from dictionary."""
        last_sync = data.get("last_sync")
        if isinstance(last_sync, str):
            last_sync = datetime.fromisoformat(last_sync)

        return cls(
            user_id=data["user_id"],
            history_id=data.get("history_id", ""),
            last_sync=last_sync,
            total_messages=data.get("total_messages", 0),
            indexed_messages=data.get("indexed_messages", 0),
            email_address=data.get("email_address", ""),
            labels_synced=data.get("labels_synced", []),
        )
