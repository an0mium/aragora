"""
RBAC Audit Logging - Track authorization decisions for compliance.

Provides comprehensive logging of all authorization decisions,
role changes, and permission modifications for audit trails.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

from .models import AuthorizationContext, AuthorizationDecision, RoleAssignment

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of authorization audit events."""

    # Permission checks
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"

    # Role management
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"
    ROLE_CREATED = "role_created"
    ROLE_DELETED = "role_deleted"
    ROLE_MODIFIED = "role_modified"

    # Session events
    SESSION_CREATED = "session_created"
    SESSION_EXPIRED = "session_expired"
    SESSION_REVOKED = "session_revoked"

    # API key events
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    API_KEY_USED = "api_key_used"

    # Admin actions
    IMPERSONATION_START = "impersonation_start"
    IMPERSONATION_END = "impersonation_end"
    POLICY_CHANGED = "policy_changed"


@dataclass
class AuditEvent:
    """
    Audit event for authorization-related actions.

    Attributes:
        id: Unique event identifier
        event_type: Type of authorization event
        timestamp: When the event occurred
        user_id: User who performed or was subject to the action
        org_id: Organization context
        actor_id: User who initiated the action (may differ from user_id)
        resource_type: Type of resource involved
        resource_id: Specific resource ID
        permission_key: Permission that was checked/modified
        decision: Outcome (allowed/denied)
        reason: Explanation for the decision
        ip_address: Request IP address
        user_agent: Request user agent
        request_id: Request trace ID
        metadata: Additional event data
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    event_type: AuditEventType = AuditEventType.PERMISSION_GRANTED
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: str | None = None
    org_id: str | None = None
    actor_id: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    permission_key: str | None = None
    decision: bool = True
    reason: str = ""
    ip_address: str | None = None
    user_agent: str | None = None
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "org_id": self.org_id,
            "actor_id": self.actor_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "permission_key": self.permission_key,
            "decision": self.decision,
            "reason": self.reason,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "request_id": self.request_id,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class AuthorizationAuditor:
    """
    Auditor for authorization events.

    Logs all authorization decisions and role changes for compliance
    and security monitoring.
    """

    def __init__(
        self,
        handlers: list[Callable[[AuditEvent], None]] | None = None,
        log_denied_only: bool = False,
        include_cached: bool = False,
    ) -> None:
        """
        Initialize the auditor.

        Args:
            handlers: List of handler functions to process events
            log_denied_only: If True, only log denied decisions
            include_cached: If True, also log cached decisions
        """
        self._handlers = handlers or []
        self._log_denied_only = log_denied_only
        self._include_cached = include_cached
        self._event_buffer: list[AuditEvent] = []
        self._buffer_size = 100

        # Add default logger handler
        self._handlers.append(self._default_log_handler)

    def log_decision(self, decision: AuthorizationDecision) -> None:
        """
        Log an authorization decision.

        Args:
            decision: The authorization decision to log
        """
        # Skip cached decisions unless configured to include
        if decision.cached and not self._include_cached:
            return

        # Skip allowed decisions if log_denied_only
        if decision.allowed and self._log_denied_only:
            return

        event = AuditEvent(
            event_type=(
                AuditEventType.PERMISSION_GRANTED
                if decision.allowed
                else AuditEventType.PERMISSION_DENIED
            ),
            timestamp=decision.checked_at,
            user_id=decision.context.user_id if decision.context else None,
            org_id=decision.context.org_id if decision.context else None,
            permission_key=decision.permission_key,
            resource_id=decision.resource_id,
            decision=decision.allowed,
            reason=decision.reason,
            ip_address=decision.context.ip_address if decision.context else None,
            user_agent=decision.context.user_agent if decision.context else None,
            request_id=decision.context.request_id if decision.context else None,
        )

        self._emit_event(event)

    def log_role_assignment(
        self,
        assignment: RoleAssignment,
        actor_id: str,
        ip_address: str | None = None,
    ) -> None:
        """Log a role assignment."""
        event = AuditEvent(
            event_type=AuditEventType.ROLE_ASSIGNED,
            user_id=assignment.user_id,
            org_id=assignment.org_id,
            actor_id=actor_id,
            resource_type="role",
            resource_id=assignment.role_id,
            decision=True,
            reason=f"Role '{assignment.role_id}' assigned to user",
            ip_address=ip_address,
            metadata={
                "assignment_id": assignment.id,
                "expires_at": assignment.expires_at.isoformat() if assignment.expires_at else None,
            },
        )

        self._emit_event(event)

    def log_role_revocation(
        self,
        user_id: str,
        role_id: str,
        org_id: str | None,
        actor_id: str,
        reason: str = "",
        ip_address: str | None = None,
    ) -> None:
        """Log a role revocation."""
        event = AuditEvent(
            event_type=AuditEventType.ROLE_REVOKED,
            user_id=user_id,
            org_id=org_id,
            actor_id=actor_id,
            resource_type="role",
            resource_id=role_id,
            decision=True,
            reason=reason or f"Role '{role_id}' revoked from user",
            ip_address=ip_address,
        )

        self._emit_event(event)

    def log_api_key_created(
        self,
        user_id: str,
        key_id: str,
        scopes: set[str],
        actor_id: str | None = None,
        ip_address: str | None = None,
    ) -> None:
        """Log API key creation."""
        event = AuditEvent(
            event_type=AuditEventType.API_KEY_CREATED,
            user_id=user_id,
            actor_id=actor_id or user_id,
            resource_type="api_key",
            resource_id=key_id,
            decision=True,
            reason="API key created",
            ip_address=ip_address,
            metadata={
                "scopes": list(scopes),
            },
        )

        self._emit_event(event)

    def log_api_key_revoked(
        self,
        user_id: str,
        key_id: str,
        actor_id: str,
        reason: str = "",
        ip_address: str | None = None,
    ) -> None:
        """Log API key revocation."""
        event = AuditEvent(
            event_type=AuditEventType.API_KEY_REVOKED,
            user_id=user_id,
            actor_id=actor_id,
            resource_type="api_key",
            resource_id=key_id,
            decision=True,
            reason=reason or "API key revoked",
            ip_address=ip_address,
        )

        self._emit_event(event)

    def log_impersonation_start(
        self,
        actor_id: str,
        target_user_id: str,
        org_id: str | None,
        reason: str,
        ip_address: str | None = None,
    ) -> None:
        """Log start of user impersonation."""
        event = AuditEvent(
            event_type=AuditEventType.IMPERSONATION_START,
            user_id=target_user_id,
            org_id=org_id,
            actor_id=actor_id,
            decision=True,
            reason=reason,
            ip_address=ip_address,
        )

        self._emit_event(event)

    def log_impersonation_end(
        self,
        actor_id: str,
        target_user_id: str,
        org_id: str | None,
        ip_address: str | None = None,
    ) -> None:
        """Log end of user impersonation."""
        event = AuditEvent(
            event_type=AuditEventType.IMPERSONATION_END,
            user_id=target_user_id,
            org_id=org_id,
            actor_id=actor_id,
            decision=True,
            reason="Impersonation session ended",
            ip_address=ip_address,
        )

        self._emit_event(event)

    def log_session_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        session_id: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
        reason: str = "",
    ) -> None:
        """Log session-related events."""
        event = AuditEvent(
            event_type=event_type,
            user_id=user_id,
            resource_type="session",
            resource_id=session_id,
            decision=True,
            reason=reason,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        self._emit_event(event)

    def add_handler(self, handler: Callable[[AuditEvent], None]) -> None:
        """Add an event handler."""
        self._handlers.append(handler)

    def remove_handler(self, handler: Callable[[AuditEvent], None]) -> None:
        """Remove an event handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)

    def flush_buffer(self) -> list[AuditEvent]:
        """Flush and return buffered events."""
        events = self._event_buffer.copy()
        self._event_buffer.clear()
        return events

    def _emit_event(self, event: AuditEvent) -> None:
        """Emit event to all handlers."""
        for handler in self._handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in audit handler: {e}")

        # Buffer for batch processing
        self._event_buffer.append(event)
        if len(self._event_buffer) >= self._buffer_size:
            self._event_buffer = self._event_buffer[-self._buffer_size:]

    def _default_log_handler(self, event: AuditEvent) -> None:
        """Default handler that logs to Python logger."""
        log_level = logging.INFO if event.decision else logging.WARNING

        logger.log(
            log_level,
            "RBAC Audit: %s | user=%s | org=%s | permission=%s | resource=%s | decision=%s | reason=%s",
            event.event_type.value,
            event.user_id,
            event.org_id,
            event.permission_key,
            event.resource_id,
            event.decision,
            event.reason,
            extra={
                "audit_event": event.to_dict(),
            },
        )


# Global auditor instance
_auditor: AuthorizationAuditor | None = None


def get_auditor() -> AuthorizationAuditor:
    """Get or create the global auditor instance."""
    global _auditor
    if _auditor is None:
        _auditor = AuthorizationAuditor()
    return _auditor


def set_auditor(auditor: AuthorizationAuditor) -> None:
    """Set the global auditor instance."""
    global _auditor
    _auditor = auditor


# Convenience functions
def log_permission_check(
    user_id: str,
    permission_key: str,
    allowed: bool,
    reason: str = "",
    resource_id: str | None = None,
    org_id: str | None = None,
    ip_address: str | None = None,
) -> None:
    """Quick function to log a permission check."""
    event = AuditEvent(
        event_type=(
            AuditEventType.PERMISSION_GRANTED
            if allowed
            else AuditEventType.PERMISSION_DENIED
        ),
        user_id=user_id,
        org_id=org_id,
        permission_key=permission_key,
        resource_id=resource_id,
        decision=allowed,
        reason=reason,
        ip_address=ip_address,
    )
    get_auditor()._emit_event(event)
