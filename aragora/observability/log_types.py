"""
Immutable Audit Log Types.

Data classes and enums for the audit logging system.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class AuditBackend(str, Enum):
    """Supported audit log backends."""

    LOCAL = "local"  # Local append-only file (good for dev/testing)
    S3_OBJECT_LOCK = "s3_object_lock"  # S3 with Object Lock (WORM compliance)
    QLDB = "qldb"  # AWS QLDB (cryptographic verification, queryable)


@dataclass
class AuditEntry:
    """A single immutable audit log entry."""

    # Unique identifier
    id: str

    # Timestamp (UTC)
    timestamp: datetime

    # Hash chain fields
    sequence_number: int
    previous_hash: str
    entry_hash: str

    # Event data
    event_type: str
    actor: str  # User ID or system identifier
    actor_type: str  # "user", "system", "agent"
    resource_type: str  # "finding", "document", "audit_session", etc.
    resource_id: str
    action: str  # "create", "update", "delete", "access", etc.

    # Additional context
    details: dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    workspace_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # Integrity
    signature: Optional[str] = None  # Optional cryptographic signature

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "sequence_number": self.sequence_number,
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
            "event_type": self.event_type,
            "actor": self.actor,
            "actor_type": self.actor_type,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "details": self.details,
            "correlation_id": self.correlation_id,
            "workspace_id": self.workspace_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditEntry:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            sequence_number=data["sequence_number"],
            previous_hash=data["previous_hash"],
            entry_hash=data["entry_hash"],
            event_type=data["event_type"],
            actor=data["actor"],
            actor_type=data.get("actor_type", "user"),
            resource_type=data["resource_type"],
            resource_id=data["resource_id"],
            action=data["action"],
            details=data.get("details", {}),
            correlation_id=data.get("correlation_id"),
            workspace_id=data.get("workspace_id"),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            signature=data.get("signature"),
        )

    def compute_hash(self) -> str:
        """Compute the hash of this entry's content."""
        content = {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "sequence_number": self.sequence_number,
            "previous_hash": self.previous_hash,
            "event_type": self.event_type,
            "actor": self.actor,
            "actor_type": self.actor_type,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "details": self.details,
            "correlation_id": self.correlation_id,
            "workspace_id": self.workspace_id,
        }
        content_bytes = json.dumps(content, sort_keys=True).encode("utf-8")
        return hashlib.sha256(content_bytes).hexdigest()


@dataclass
class DailyAnchor:
    """Daily hash anchor for external verification."""

    date: str  # YYYY-MM-DD
    first_sequence: int
    last_sequence: int
    entry_count: int
    merkle_root: str  # Root of Merkle tree for day's entries
    chain_hash: str  # Hash of last entry in day
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date,
            "first_sequence": self.first_sequence,
            "last_sequence": self.last_sequence,
            "entry_count": self.entry_count,
            "merkle_root": self.merkle_root,
            "chain_hash": self.chain_hash,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class VerificationResult:
    """Result of integrity verification."""

    is_valid: bool
    entries_checked: int
    errors: list[str]
    warnings: list[str]
    first_error_sequence: Optional[int] = None
    verification_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "entries_checked": self.entries_checked,
            "errors": self.errors,
            "warnings": self.warnings,
            "first_error_sequence": self.first_error_sequence,
            "verification_time_ms": self.verification_time_ms,
        }
