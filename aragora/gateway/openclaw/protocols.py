"""
Protocol definitions for OpenClaw dependency injection.

Provides typing protocols for RBAC checking, audit logging, and
approval gate checking. These allow the adapter to accept any
implementation conforming to the protocol interface.
"""

from __future__ import annotations

from typing import Any, Protocol


class RBACCheckerProtocol(Protocol):
    """Protocol for RBAC permission checking."""

    def check_permission(
        self,
        actor_id: str,
        permission: str,
        resource_id: str | None = None,
    ) -> bool:
        """Check if actor has permission."""
        ...

    async def check_permission_async(
        self,
        actor_id: str,
        permission: str,
        resource_id: str | None = None,
    ) -> bool:
        """Async check if actor has permission."""
        ...


class AuditLoggerProtocol(Protocol):
    """Protocol for audit logging."""

    def log(
        self,
        event_type: str,
        actor_id: str,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
        severity: str = "info",
    ) -> None:
        """Log an audit event."""
        ...

    async def log_async(
        self,
        event_type: str,
        actor_id: str,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
        severity: str = "info",
    ) -> None:
        """Async log an audit event."""
        ...


class ApprovalGateProtocol(Protocol):
    """Protocol for approval gate checking."""

    async def check_approval(
        self,
        gate: str,
        actor_id: str,
        resource_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Check if actor has approval for gate. Returns (approved, reason)."""
        ...
