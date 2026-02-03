"""
In-memory storage for OpenClaw Gateway.

Stability: STABLE

Contains:
- OpenClawGatewayStore - in-memory data store
- Global store instance management
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from aragora.server.handlers.openclaw.models import (
    Action,
    ActionStatus,
    AuditEntry,
    Credential,
    CredentialType,
    Session,
    SessionStatus,
)


class OpenClawGatewayStore:
    """In-memory store for OpenClaw gateway data.

    In production, this would be replaced with a persistent storage backend
    (PostgreSQL, Redis, etc.).
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._actions: dict[str, Action] = {}
        self._credentials: dict[str, Credential] = {}
        self._credential_secrets: dict[str, str] = {}  # Stored separately
        self._audit_log: list[AuditEntry] = []

    # Session methods
    def create_session(
        self,
        user_id: str,
        tenant_id: str | None = None,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Create a new session."""
        now = datetime.now(timezone.utc)
        session = Session(
            id=str(uuid.uuid4()),
            user_id=user_id,
            tenant_id=tenant_id,
            status=SessionStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            last_activity_at=now,
            config=config or {},
            metadata=metadata or {},
        )
        self._sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Get session by ID."""
        return self._sessions.get(session_id)

    def list_sessions(
        self,
        user_id: str | None = None,
        tenant_id: str | None = None,
        status: SessionStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Session], int]:
        """List sessions with optional filtering."""
        sessions = list(self._sessions.values())

        # Apply filters
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        if tenant_id:
            sessions = [s for s in sessions if s.tenant_id == tenant_id]
        if status:
            sessions = [s for s in sessions if s.status == status]

        # Sort by created_at descending
        sessions.sort(key=lambda s: s.created_at, reverse=True)

        total = len(sessions)
        return sessions[offset : offset + limit], total

    def update_session_status(self, session_id: str, status: SessionStatus) -> Session | None:
        """Update session status."""
        session = self._sessions.get(session_id)
        if session:
            session.status = status
            session.updated_at = datetime.now(timezone.utc)
        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    # Action methods
    def create_action(
        self,
        session_id: str,
        action_type: str,
        input_data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> Action:
        """Create a new action."""
        now = datetime.now(timezone.utc)
        action = Action(
            id=str(uuid.uuid4()),
            session_id=session_id,
            action_type=action_type,
            status=ActionStatus.PENDING,
            input_data=input_data,
            output_data=None,
            error=None,
            created_at=now,
            started_at=None,
            completed_at=None,
            metadata=metadata or {},
        )
        self._actions[action.id] = action
        return action

    def get_action(self, action_id: str) -> Action | None:
        """Get action by ID."""
        return self._actions.get(action_id)

    def update_action(
        self,
        action_id: str,
        status: ActionStatus | None = None,
        output_data: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> Action | None:
        """Update action state."""
        action = self._actions.get(action_id)
        if action:
            now = datetime.now(timezone.utc)
            if status:
                action.status = status
                if status == ActionStatus.RUNNING and not action.started_at:
                    action.started_at = now
                elif status in (
                    ActionStatus.COMPLETED,
                    ActionStatus.FAILED,
                    ActionStatus.CANCELLED,
                ):
                    action.completed_at = now
            if output_data is not None:
                action.output_data = output_data
            if error is not None:
                action.error = error
        return action

    # Credential methods
    def store_credential(
        self,
        name: str,
        credential_type: CredentialType,
        secret_value: str,
        user_id: str,
        tenant_id: str | None = None,
        expires_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Credential:
        """Store a new credential."""
        now = datetime.now(timezone.utc)
        credential = Credential(
            id=str(uuid.uuid4()),
            name=name,
            credential_type=credential_type,
            user_id=user_id,
            tenant_id=tenant_id,
            created_at=now,
            updated_at=now,
            last_rotated_at=None,
            expires_at=expires_at,
            metadata=metadata or {},
        )
        self._credentials[credential.id] = credential
        self._credential_secrets[credential.id] = secret_value
        return credential

    def get_credential(self, credential_id: str) -> Credential | None:
        """Get credential metadata by ID (not the secret)."""
        return self._credentials.get(credential_id)

    def list_credentials(
        self,
        user_id: str | None = None,
        tenant_id: str | None = None,
        credential_type: CredentialType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Credential], int]:
        """List credentials with optional filtering (no secret values)."""
        credentials = list(self._credentials.values())

        # Apply filters
        if user_id:
            credentials = [c for c in credentials if c.user_id == user_id]
        if tenant_id:
            credentials = [c for c in credentials if c.tenant_id == tenant_id]
        if credential_type:
            credentials = [c for c in credentials if c.credential_type == credential_type]

        # Sort by created_at descending
        credentials.sort(key=lambda c: c.created_at, reverse=True)

        total = len(credentials)
        return credentials[offset : offset + limit], total

    def delete_credential(self, credential_id: str) -> bool:
        """Delete a credential."""
        if credential_id in self._credentials:
            del self._credentials[credential_id]
            del self._credential_secrets[credential_id]
            return True
        return False

    def rotate_credential(self, credential_id: str, new_secret_value: str) -> Credential | None:
        """Rotate a credential's secret value."""
        credential = self._credentials.get(credential_id)
        if credential:
            now = datetime.now(timezone.utc)
            credential.last_rotated_at = now
            credential.updated_at = now
            self._credential_secrets[credential_id] = new_secret_value
        return credential

    # Audit methods
    def add_audit_entry(
        self,
        action: str,
        actor_id: str,
        resource_type: str,
        resource_id: str | None = None,
        result: str = "success",
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Add an audit log entry."""
        entry = AuditEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            action=action,
            actor_id=actor_id,
            resource_type=resource_type,
            resource_id=resource_id,
            result=result,
            details=details or {},
        )
        self._audit_log.append(entry)
        # Keep only last 10000 entries
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-10000:]
        return entry

    def get_audit_log(
        self,
        action: str | None = None,
        actor_id: str | None = None,
        resource_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[AuditEntry], int]:
        """Get audit log entries with optional filtering."""
        entries = self._audit_log.copy()

        # Apply filters
        if action:
            entries = [e for e in entries if e.action == action]
        if actor_id:
            entries = [e for e in entries if e.actor_id == actor_id]
        if resource_type:
            entries = [e for e in entries if e.resource_type == resource_type]

        # Sort by timestamp descending (most recent first)
        entries.sort(key=lambda e: e.timestamp, reverse=True)

        total = len(entries)
        return entries[offset : offset + limit], total

    # Metrics
    def get_metrics(self) -> dict[str, Any]:
        """Get gateway metrics."""
        active_sessions = sum(
            1 for s in self._sessions.values() if s.status == SessionStatus.ACTIVE
        )
        pending_actions = sum(1 for a in self._actions.values() if a.status == ActionStatus.PENDING)
        running_actions = sum(1 for a in self._actions.values() if a.status == ActionStatus.RUNNING)

        return {
            "sessions": {
                "total": len(self._sessions),
                "active": active_sessions,
                "by_status": {
                    status.value: sum(1 for s in self._sessions.values() if s.status == status)
                    for status in SessionStatus
                },
            },
            "actions": {
                "total": len(self._actions),
                "pending": pending_actions,
                "running": running_actions,
                "by_status": {
                    status.value: sum(1 for a in self._actions.values() if a.status == status)
                    for status in ActionStatus
                },
            },
            "credentials": {
                "total": len(self._credentials),
                "by_type": {
                    ctype.value: sum(
                        1 for c in self._credentials.values() if c.credential_type == ctype
                    )
                    for ctype in CredentialType
                },
            },
            "audit_log_entries": len(self._audit_log),
        }


# Global store instance
_store: OpenClawGatewayStore | None = None


def _get_store() -> OpenClawGatewayStore:
    """Get or create the global store instance."""
    global _store
    if _store is None:
        _store = OpenClawGatewayStore()
    return _store


__all__ = [
    "OpenClawGatewayStore",
    "_get_store",
]
