"""
Break-Glass Emergency Access.

Implements emergency access override mechanism for critical situations:
- Time-limited elevated access
- Extended audit logging during emergency
- Automatic notification to security team
- Post-incident review requirements

Usage:
    from aragora.rbac.emergency import BreakGlassAccess

    emergency = BreakGlassAccess()

    # Activate break-glass access
    access_id = await emergency.activate(
        user_id="user-123",
        reason="Production incident - database corruption",
        duration_minutes=60,
    )

    # Check if user has emergency access
    if await emergency.is_active(user_id="user-123"):
        # User has elevated permissions

    # Deactivate when done
    await emergency.deactivate(access_id)

IMPORTANT: Break-glass access should be used only for genuine emergencies.
All actions during emergency access are logged with enhanced audit detail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class EmergencyAccessStatus(str, Enum):
    """Status of emergency access."""

    ACTIVE = "active"
    EXPIRED = "expired"
    DEACTIVATED = "deactivated"
    REVOKED = "revoked"  # Forcibly revoked by security


@dataclass
class EmergencyAccessRecord:
    """Record of break-glass access activation."""

    id: str
    user_id: str
    reason: str
    status: EmergencyAccessStatus
    activated_at: datetime
    expires_at: datetime
    deactivated_at: Optional[datetime] = None
    deactivated_by: Optional[str] = None  # User who deactivated
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    actions_taken: list[dict[str, Any]] = field(default_factory=list)
    review_required: bool = True
    review_completed: bool = False
    review_notes: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Check if access is currently active."""
        if self.status != EmergencyAccessStatus.ACTIVE:
            return False
        return datetime.now(timezone.utc) < self.expires_at

    @property
    def duration_minutes(self) -> int:
        """Duration of access in minutes."""
        return int((self.expires_at - self.activated_at).total_seconds() / 60)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "reason": self.reason,
            "status": self.status.value,
            "activated_at": self.activated_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "deactivated_at": self.deactivated_at.isoformat() if self.deactivated_at else None,
            "deactivated_by": self.deactivated_by,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "actions_count": len(self.actions_taken),
            "review_required": self.review_required,
            "review_completed": self.review_completed,
            "duration_minutes": self.duration_minutes,
            "is_active": self.is_active,
            "metadata": self.metadata,
        }


class BreakGlassAccess:
    """
    Emergency access management for break-glass scenarios.

    Features:
    - Time-limited elevated access (default 1 hour, max 24 hours)
    - Enhanced audit logging for all actions
    - Automatic security team notification
    - Post-incident review workflow
    - IP and user-agent tracking

    Security:
    - MFA should be verified before activation (not enforced here)
    - All actions during emergency are logged with break-glass flag
    - Automatic expiration prevents forgotten elevated access
    - Security team can revoke access at any time
    """

    # Access duration limits (minutes)
    DEFAULT_DURATION_MINUTES = 60  # 1 hour
    MAX_DURATION_MINUTES = 24 * 60  # 24 hours
    MIN_DURATION_MINUTES = 15  # 15 minutes

    # Permissions granted during emergency
    EMERGENCY_PERMISSIONS = [
        "admin",
        "debates:*",
        "agents:*",
        "memory:*",
        "knowledge:*",
        "workflows:*",
        "backups:*",
        "audit_log:read",
    ]

    def __init__(self):
        """Initialize break-glass access manager."""
        self._active_records: dict[str, EmergencyAccessRecord] = {}
        self._by_user: dict[str, list[str]] = {}  # user_id -> record_ids
        self._all_records: dict[str, EmergencyAccessRecord] = {}  # Historical

    async def activate(
        self,
        user_id: str,
        reason: str,
        duration_minutes: int = DEFAULT_DURATION_MINUTES,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Activate break-glass emergency access.

        Args:
            user_id: User activating emergency access
            reason: Reason for emergency access (required)
            duration_minutes: How long access should last
            ip_address: Client IP address for audit
            user_agent: Client user agent for audit
            metadata: Additional metadata

        Returns:
            Access ID for the emergency session

        Raises:
            ValueError: If reason is empty or duration invalid
        """
        # Validate inputs
        if not reason or len(reason.strip()) < 10:
            raise ValueError("Reason must be at least 10 characters")

        if duration_minutes < self.MIN_DURATION_MINUTES:
            raise ValueError(f"Duration must be at least {self.MIN_DURATION_MINUTES} minutes")

        if duration_minutes > self.MAX_DURATION_MINUTES:
            raise ValueError(
                f"Duration cannot exceed {self.MAX_DURATION_MINUTES} minutes (24 hours)"
            )

        # Check if user already has active access
        existing = await self.get_active_access(user_id)
        if existing:
            logger.warning(f"User {user_id} already has active emergency access: {existing.id}")
            # Extend existing access instead of creating new
            existing.expires_at = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
            return existing.id

        # Create access record
        now = datetime.now(timezone.utc)
        record = EmergencyAccessRecord(
            id=f"emerg-{uuid4().hex[:12]}",
            user_id=user_id,
            reason=reason,
            status=EmergencyAccessStatus.ACTIVE,
            activated_at=now,
            expires_at=now + timedelta(minutes=duration_minutes),
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {},
        )

        # Store record
        self._active_records[record.id] = record
        self._all_records[record.id] = record

        if user_id not in self._by_user:
            self._by_user[user_id] = []
        self._by_user[user_id].append(record.id)

        logger.warning(
            f"BREAK-GLASS ACTIVATED: user={user_id}, id={record.id}, "
            f"duration={duration_minutes}min, reason={reason[:50]}..."
        )

        # Notify security team
        await self._notify_security_team(record, "activated")

        # Audit log
        await self._audit_log(
            "break_glass_activated",
            record_id=record.id,
            user_id=user_id,
            reason=reason,
            duration_minutes=duration_minutes,
            ip_address=ip_address,
        )

        return record.id

    async def deactivate(
        self,
        access_id: str,
        deactivated_by: Optional[str] = None,
    ) -> EmergencyAccessRecord:
        """
        Deactivate break-glass access.

        Args:
            access_id: Emergency access ID to deactivate
            deactivated_by: User who deactivated (if not the same user)

        Returns:
            Updated EmergencyAccessRecord
        """
        record = self._active_records.get(access_id)
        if not record:
            record = self._all_records.get(access_id)
            if not record:
                raise ValueError(f"Access record not found: {access_id}")
            if record.status != EmergencyAccessStatus.ACTIVE:
                raise ValueError(f"Access is not active: {record.status.value}")

        record.status = EmergencyAccessStatus.DEACTIVATED
        record.deactivated_at = datetime.now(timezone.utc)
        record.deactivated_by = deactivated_by or record.user_id

        # Remove from active
        self._active_records.pop(access_id, None)

        logger.info(
            f"BREAK-GLASS DEACTIVATED: id={access_id}, "
            f"user={record.user_id}, by={record.deactivated_by}"
        )

        # Notify security team
        await self._notify_security_team(record, "deactivated")

        # Audit log
        await self._audit_log(
            "break_glass_deactivated",
            record_id=access_id,
            user_id=record.user_id,
            deactivated_by=record.deactivated_by,
            actions_count=len(record.actions_taken),
        )

        return record

    async def revoke(
        self,
        access_id: str,
        revoked_by: str,
        reason: str,
    ) -> EmergencyAccessRecord:
        """
        Revoke break-glass access (security team action).

        Args:
            access_id: Emergency access ID to revoke
            revoked_by: Security team member revoking access
            reason: Reason for revocation

        Returns:
            Updated EmergencyAccessRecord
        """
        record = self._active_records.get(access_id)
        if not record:
            record = self._all_records.get(access_id)
            if not record:
                raise ValueError(f"Access record not found: {access_id}")

        record.status = EmergencyAccessStatus.REVOKED
        record.deactivated_at = datetime.now(timezone.utc)
        record.deactivated_by = revoked_by
        record.metadata["revocation_reason"] = reason

        # Remove from active
        self._active_records.pop(access_id, None)

        logger.warning(
            f"BREAK-GLASS REVOKED: id={access_id}, "
            f"user={record.user_id}, by={revoked_by}, reason={reason}"
        )

        # Audit log (critical severity)
        await self._audit_log(
            "break_glass_revoked",
            record_id=access_id,
            user_id=record.user_id,
            revoked_by=revoked_by,
            reason=reason,
            severity="critical",
        )

        return record

    async def is_active(self, user_id: str) -> bool:
        """
        Check if a user has active break-glass access.

        Args:
            user_id: User to check

        Returns:
            True if user has active emergency access
        """
        record = await self.get_active_access(user_id)
        return record is not None and record.is_active

    async def get_active_access(self, user_id: str) -> Optional[EmergencyAccessRecord]:
        """
        Get active emergency access for a user.

        Args:
            user_id: User to check

        Returns:
            EmergencyAccessRecord if active, None otherwise
        """
        record_ids = self._by_user.get(user_id, [])

        for rid in reversed(record_ids):  # Most recent first
            record = self._active_records.get(rid)
            if record and record.is_active:
                return record

        return None

    async def record_action(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Record an action taken during emergency access.

        Args:
            user_id: User taking action
            action: Action performed
            resource_type: Type of resource affected
            resource_id: Specific resource ID
            details: Additional details
        """
        record = await self.get_active_access(user_id)
        if not record:
            return  # Not in emergency mode

        action_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "details": details or {},
        }

        record.actions_taken.append(action_record)

        # Audit log with break-glass flag
        await self._audit_log(
            "break_glass_action",
            record_id=record.id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            is_break_glass=True,
        )

    async def expire_old_access(self) -> int:
        """
        Expire access records that have passed their expiration.

        Returns:
            Number of records expired
        """
        count = 0
        now = datetime.now(timezone.utc)
        to_remove = []

        for access_id, record in self._active_records.items():
            if record.expires_at < now:
                record.status = EmergencyAccessStatus.EXPIRED
                record.deactivated_at = now
                to_remove.append(access_id)
                count += 1

                logger.info(f"BREAK-GLASS EXPIRED: id={access_id}, user={record.user_id}")

                await self._audit_log(
                    "break_glass_expired",
                    record_id=access_id,
                    user_id=record.user_id,
                    actions_count=len(record.actions_taken),
                )

                # Notify security
                await self._notify_security_team(record, "expired")

        for access_id in to_remove:
            self._active_records.pop(access_id, None)

        return count

    async def get_history(
        self,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[EmergencyAccessRecord]:
        """
        Get break-glass access history.

        Args:
            user_id: Filter by user (optional)
            limit: Maximum results

        Returns:
            List of EmergencyAccessRecords
        """
        if user_id:
            record_ids = self._by_user.get(user_id, [])
            records = [self._all_records[rid] for rid in record_ids if rid in self._all_records]
        else:
            records = list(self._all_records.values())

        # Sort by activation time (most recent first)
        records.sort(key=lambda r: r.activated_at, reverse=True)

        return records[:limit]

    async def complete_review(
        self,
        access_id: str,
        reviewer_id: str,
        notes: str,
    ) -> EmergencyAccessRecord:
        """
        Complete post-incident review for an emergency access session.

        Args:
            access_id: Emergency access ID
            reviewer_id: Person completing the review
            notes: Review notes

        Returns:
            Updated EmergencyAccessRecord
        """
        record = self._all_records.get(access_id)
        if not record:
            raise ValueError(f"Access record not found: {access_id}")

        if record.status == EmergencyAccessStatus.ACTIVE:
            raise ValueError("Cannot review active access - deactivate first")

        record.review_completed = True
        record.review_notes = notes
        record.metadata["reviewed_by"] = reviewer_id
        record.metadata["reviewed_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"Break-glass review completed: id={access_id}, by={reviewer_id}")

        await self._audit_log(
            "break_glass_reviewed",
            record_id=access_id,
            reviewer_id=reviewer_id,
            notes=notes[:200] if notes else None,
        )

        return record

    async def _notify_security_team(
        self,
        record: EmergencyAccessRecord,
        event: str,
    ) -> None:
        """
        Notify security team of break-glass event.

        This should integrate with notification system (Slack, PagerDuty, email).
        """
        try:
            from aragora.control_plane.notifications import send_security_notification

            await send_security_notification(
                title=f"Break-Glass Access {event.title()}",
                message=(
                    f"User: {record.user_id}\n"
                    f"Access ID: {record.id}\n"
                    f"Reason: {record.reason}\n"
                    f"Duration: {record.duration_minutes} minutes\n"
                    f"IP: {record.ip_address or 'Unknown'}"
                ),
                severity="high" if event == "activated" else "medium",
                metadata=record.to_dict(),
            )
        except ImportError:
            logger.info(
                f"Security notification (not sent): break-glass {event} for {record.user_id}"
            )

    async def _audit_log(self, event_type: str, **kwargs) -> None:
        """Log an audit event for break-glass actions."""
        try:
            from aragora.rbac.audit import AuthorizationAuditor

            auditor = AuthorizationAuditor()
            await auditor.log_event(
                event_type=event_type,
                details=kwargs,
                category="break_glass",
            )
        except ImportError:
            logger.info(f"Break-glass audit: {event_type} - {kwargs}")


# Singleton instance
_break_glass: Optional[BreakGlassAccess] = None


def get_break_glass_access() -> BreakGlassAccess:
    """Get the global BreakGlassAccess instance."""
    global _break_glass
    if _break_glass is None:
        _break_glass = BreakGlassAccess()
    return _break_glass


__all__ = [
    "EmergencyAccessStatus",
    "EmergencyAccessRecord",
    "BreakGlassAccess",
    "get_break_glass_access",
]
