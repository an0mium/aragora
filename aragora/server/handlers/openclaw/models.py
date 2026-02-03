"""
Data models for OpenClaw Gateway.

Stability: STABLE

Contains:
- Session, Action, Credential, AuditEntry dataclasses
- SessionStatus, ActionStatus, CredentialType enums
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# =============================================================================
# Enums
# =============================================================================


class SessionStatus(Enum):
    """OpenClaw session status."""

    ACTIVE = "active"
    IDLE = "idle"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"


class ActionStatus(Enum):
    """OpenClaw action execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class CredentialType(Enum):
    """Credential types supported by the gateway."""

    API_KEY = "api_key"
    OAUTH_TOKEN = "oauth_token"
    PASSWORD = "password"
    CERTIFICATE = "certificate"
    SSH_KEY = "ssh_key"
    SERVICE_ACCOUNT = "service_account"


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class Session:
    """OpenClaw session data."""

    id: str
    user_id: str
    tenant_id: str | None
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_activity_at": self.last_activity_at.isoformat(),
            "config": self.config,
            "metadata": self.metadata,
        }


@dataclass
class Action:
    """OpenClaw action data."""

    id: str
    session_id: str
    action_type: str
    status: ActionStatus
    input_data: dict[str, Any]
    output_data: dict[str, Any] | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "action_type": self.action_type,
            "status": self.status.value,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }


@dataclass
class Credential:
    """OpenClaw credential metadata (never includes actual secret values)."""

    id: str
    name: str
    credential_type: CredentialType
    user_id: str
    tenant_id: str | None
    created_at: datetime
    updated_at: datetime
    last_rotated_at: datetime | None
    expires_at: datetime | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization (excludes secret)."""
        return {
            "id": self.id,
            "name": self.name,
            "credential_type": self.credential_type.value,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_rotated_at": (self.last_rotated_at.isoformat() if self.last_rotated_at else None),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }


@dataclass
class AuditEntry:
    """Audit log entry for OpenClaw gateway operations."""

    id: str
    timestamp: datetime
    action: str
    actor_id: str
    resource_type: str
    resource_id: str | None
    result: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "actor_id": self.actor_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "result": self.result,
            "details": self.details,
        }


__all__ = [
    # Enums
    "SessionStatus",
    "ActionStatus",
    "CredentialType",
    # Data models
    "Session",
    "Action",
    "Credential",
    "AuditEntry",
]
