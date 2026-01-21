"""
Impersonation controls and audit logging.

Provides secure admin impersonation with:
1. Full audit trail of all impersonation actions
2. Time-limited impersonation sessions
3. User notification of active impersonation
4. 2FA requirements for admin impersonation
"""

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Global recovery state
_sessions_recovered: bool = False


@dataclass
class ImpersonationSession:
    """Active impersonation session."""

    session_id: str
    admin_user_id: str
    admin_email: str
    target_user_id: str
    target_email: str
    reason: str
    started_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    actions_performed: int = 0

    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now(timezone.utc) > self.expires_at

    def to_audit_dict(self) -> Dict[str, Any]:
        """Convert to audit log format."""
        return {
            "session_id": self.session_id,
            "admin_user_id": self.admin_user_id,
            "admin_email": self.admin_email,
            "target_user_id": self.target_user_id,
            "target_email": self.target_email,
            "reason": self.reason,
            "started_at": self.started_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "actions_performed": self.actions_performed,
        }


@dataclass
class ImpersonationAuditEntry:
    """Audit log entry for impersonation action."""

    timestamp: datetime
    event_type: str  # start, action, end, timeout, denied
    session_id: Optional[str]
    admin_user_id: str
    target_user_id: Optional[str]
    reason: Optional[str]
    action_details: Optional[Dict[str, Any]]
    ip_address: str
    user_agent: str
    success: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to storage format."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "session_id": self.session_id,
            "admin_user_id": self.admin_user_id,
            "target_user_id": self.target_user_id,
            "reason": self.reason,
            "action_details": self.action_details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "success": self.success,
            "error_message": self.error_message,
        }


class ImpersonationManager:
    """
    Manages admin impersonation with security controls and audit logging.

    Security features:
    - All impersonation actions are logged
    - Sessions are time-limited (default 1 hour max)
    - Target users are notified of impersonation
    - 2FA required for impersonating admins
    - Reason required for audit trail
    """

    # Maximum session duration
    MAX_SESSION_DURATION = timedelta(hours=1)

    # Default session duration
    DEFAULT_SESSION_DURATION = timedelta(minutes=30)

    def __init__(
        self,
        audit_callback: Optional[Callable[[ImpersonationAuditEntry], None]] = None,
        notification_callback: Optional[Callable[[str, str, str], None]] = None,
        require_2fa_for_admin_targets: bool = True,
        max_concurrent_sessions: int = 3,
    ):
        """
        Initialize impersonation manager.

        Args:
            audit_callback: Function to persist audit entries (receives ImpersonationAuditEntry)
            notification_callback: Function to notify target user (user_id, admin_email, reason)
            require_2fa_for_admin_targets: Require 2FA when impersonating admin users
            max_concurrent_sessions: Maximum concurrent impersonation sessions per admin
        """
        self._audit_callback = audit_callback
        self._notification_callback = notification_callback
        self._require_2fa_for_admin_targets = require_2fa_for_admin_targets
        self._max_concurrent_sessions = max_concurrent_sessions

        # Active sessions: session_id -> ImpersonationSession
        self._sessions: Dict[str, ImpersonationSession] = {}

        # Admin -> list of active session_ids
        self._admin_sessions: Dict[str, List[str]] = {}

        # In-memory audit log (backup if callback not configured)
        self._audit_log: List[ImpersonationAuditEntry] = []

        # Optional persistent store
        self._store = None
        self._use_persistence = True  # Can be disabled for testing

    def _get_store(self):
        """Lazily get the impersonation store."""
        if self._store is None and self._use_persistence:
            try:
                from aragora.storage.impersonation_store import get_impersonation_store

                self._store = get_impersonation_store()
            except Exception as e:
                logger.warning(f"Failed to get impersonation store: {e}")
        return self._store

    def _generate_session_id(self, admin_id: str, target_id: str) -> str:
        """Generate unique session ID."""
        timestamp = str(time.time_ns())
        data = f"{admin_id}:{target_id}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _log_audit(self, entry: ImpersonationAuditEntry) -> None:
        """Log audit entry via callback, in-memory, and persistent store."""
        self._audit_log.append(entry)

        # Keep in-memory log bounded
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-5000:]

        # Persist to store
        store = self._get_store()
        if store:
            try:
                audit_id = str(uuid.uuid4())
                store.save_audit_entry(
                    audit_id=audit_id,
                    timestamp=entry.timestamp,
                    event_type=entry.event_type,
                    admin_user_id=entry.admin_user_id,
                    ip_address=entry.ip_address,
                    user_agent=entry.user_agent,
                    success=entry.success,
                    session_id=entry.session_id,
                    target_user_id=entry.target_user_id,
                    reason=entry.reason,
                    action_details=entry.action_details,
                    error_message=entry.error_message,
                )
            except Exception as e:
                logger.error(f"Failed to persist audit entry to store: {e}")

        if self._audit_callback:
            try:
                self._audit_callback(entry)
            except Exception as e:
                logger.error(f"Failed to persist audit entry via callback: {e}")

        # Always log to structured logger
        logger.info(
            f"[IMPERSONATION AUDIT] {entry.event_type}: "
            f"admin={entry.admin_user_id} target={entry.target_user_id} "
            f"session={entry.session_id} success={entry.success} "
            f"ip={entry.ip_address}"
        )

    def _notify_target(self, session: ImpersonationSession) -> None:
        """Notify target user of impersonation."""
        if self._notification_callback:
            try:
                self._notification_callback(
                    session.target_user_id,
                    session.admin_email,
                    session.reason,
                )
            except Exception as e:
                logger.error(f"Failed to notify target user: {e}")

    def start_impersonation(
        self,
        admin_user_id: str,
        admin_email: str,
        admin_roles: List[str],
        target_user_id: str,
        target_email: str,
        target_roles: List[str],
        reason: str,
        ip_address: str,
        user_agent: str,
        has_2fa: bool = False,
        duration: Optional[timedelta] = None,
    ) -> tuple[Optional[ImpersonationSession], str]:
        """
        Start an impersonation session.

        Args:
            admin_user_id: ID of admin initiating impersonation
            admin_email: Email of admin
            admin_roles: Roles of admin user
            target_user_id: ID of user to impersonate
            target_email: Email of target user
            target_roles: Roles of target user
            reason: Required justification for impersonation
            ip_address: Request IP address
            user_agent: Request user agent
            has_2fa: Whether admin has completed 2FA
            duration: Session duration (capped at MAX_SESSION_DURATION)

        Returns:
            Tuple of (session or None, error_message or success_message)
        """
        # Validate reason
        if not reason or len(reason.strip()) < 10:
            entry = ImpersonationAuditEntry(
                timestamp=datetime.now(timezone.utc),
                event_type="denied",
                session_id=None,
                admin_user_id=admin_user_id,
                target_user_id=target_user_id,
                reason=reason,
                action_details=None,
                ip_address=ip_address,
                user_agent=user_agent,
                success=False,
                error_message="Reason must be at least 10 characters",
            )
            self._log_audit(entry)
            return None, "Impersonation reason must be at least 10 characters"

        # Prevent self-impersonation
        if admin_user_id == target_user_id:
            entry = ImpersonationAuditEntry(
                timestamp=datetime.now(timezone.utc),
                event_type="denied",
                session_id=None,
                admin_user_id=admin_user_id,
                target_user_id=target_user_id,
                reason=reason,
                action_details=None,
                ip_address=ip_address,
                user_agent=user_agent,
                success=False,
                error_message="Cannot impersonate yourself",
            )
            self._log_audit(entry)
            return None, "Cannot impersonate yourself"

        # Check if target is admin and 2FA required
        target_is_admin = "admin" in target_roles or "owner" in target_roles
        if target_is_admin and self._require_2fa_for_admin_targets and not has_2fa:
            entry = ImpersonationAuditEntry(
                timestamp=datetime.now(timezone.utc),
                event_type="denied",
                session_id=None,
                admin_user_id=admin_user_id,
                target_user_id=target_user_id,
                reason=reason,
                action_details={"target_is_admin": True, "has_2fa": False},
                ip_address=ip_address,
                user_agent=user_agent,
                success=False,
                error_message="2FA required to impersonate admin users",
            )
            self._log_audit(entry)
            return None, "2FA verification required to impersonate admin users"

        # Check concurrent session limit
        admin_session_ids = self._admin_sessions.get(admin_user_id, [])
        active_sessions = [
            sid
            for sid in admin_session_ids
            if sid in self._sessions and not self._sessions[sid].is_expired()
        ]

        if len(active_sessions) >= self._max_concurrent_sessions:
            entry = ImpersonationAuditEntry(
                timestamp=datetime.now(timezone.utc),
                event_type="denied",
                session_id=None,
                admin_user_id=admin_user_id,
                target_user_id=target_user_id,
                reason=reason,
                action_details={"active_sessions": len(active_sessions)},
                ip_address=ip_address,
                user_agent=user_agent,
                success=False,
                error_message=f"Maximum concurrent sessions ({self._max_concurrent_sessions}) reached",
            )
            self._log_audit(entry)
            return (
                None,
                f"Maximum concurrent impersonation sessions ({self._max_concurrent_sessions}) reached",
            )

        # Calculate duration
        effective_duration = duration or self.DEFAULT_SESSION_DURATION
        if effective_duration > self.MAX_SESSION_DURATION:
            effective_duration = self.MAX_SESSION_DURATION

        # Create session
        session_id = self._generate_session_id(admin_user_id, target_user_id)
        now = datetime.now(timezone.utc)
        session = ImpersonationSession(
            session_id=session_id,
            admin_user_id=admin_user_id,
            admin_email=admin_email,
            target_user_id=target_user_id,
            target_email=target_email,
            reason=reason,
            started_at=now,
            expires_at=now + effective_duration,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        # Store session in memory
        self._sessions[session_id] = session
        if admin_user_id not in self._admin_sessions:
            self._admin_sessions[admin_user_id] = []
        self._admin_sessions[admin_user_id].append(session_id)

        # Persist session to store
        store = self._get_store()
        if store:
            try:
                store.save_session(
                    session_id=session_id,
                    admin_user_id=admin_user_id,
                    admin_email=admin_email,
                    target_user_id=target_user_id,
                    target_email=target_email,
                    reason=reason,
                    started_at=now,
                    expires_at=session.expires_at,
                    ip_address=ip_address,
                    user_agent=user_agent,
                )
            except Exception as e:
                logger.error(f"Failed to persist session to store: {e}")

        # Log audit
        entry = ImpersonationAuditEntry(
            timestamp=now,
            event_type="start",
            session_id=session_id,
            admin_user_id=admin_user_id,
            target_user_id=target_user_id,
            reason=reason,
            action_details={
                "duration_seconds": effective_duration.total_seconds(),
                "target_is_admin": target_is_admin,
            },
            ip_address=ip_address,
            user_agent=user_agent,
            success=True,
        )
        self._log_audit(entry)

        # Notify target user
        self._notify_target(session)

        return session, f"Impersonation session started (expires in {effective_duration})"

    def end_impersonation(
        self,
        session_id: str,
        admin_user_id: str,
        ip_address: str,
        user_agent: str,
    ) -> tuple[bool, str]:
        """
        End an impersonation session.

        Args:
            session_id: Session to end
            admin_user_id: Admin ending the session (must match session admin)
            ip_address: Request IP address
            user_agent: Request user agent

        Returns:
            Tuple of (success, message)
        """
        session = self._sessions.get(session_id)
        if not session:
            return False, "Session not found"

        if session.admin_user_id != admin_user_id:
            entry = ImpersonationAuditEntry(
                timestamp=datetime.now(timezone.utc),
                event_type="denied",
                session_id=session_id,
                admin_user_id=admin_user_id,
                target_user_id=session.target_user_id,
                reason=None,
                action_details={"attempted_by": admin_user_id},
                ip_address=ip_address,
                user_agent=user_agent,
                success=False,
                error_message="Only session owner can end session",
            )
            self._log_audit(entry)
            return False, "Only the admin who started the session can end it"

        # Log end
        entry = ImpersonationAuditEntry(
            timestamp=datetime.now(timezone.utc),
            event_type="end",
            session_id=session_id,
            admin_user_id=admin_user_id,
            target_user_id=session.target_user_id,
            reason=session.reason,
            action_details={"actions_performed": session.actions_performed},
            ip_address=ip_address,
            user_agent=user_agent,
            success=True,
        )
        self._log_audit(entry)

        # Mark session as ended in store
        store = self._get_store()
        if store:
            try:
                store.end_session(
                    session_id=session_id,
                    ended_by="admin",
                    actions_performed=session.actions_performed,
                )
            except Exception as e:
                logger.error(f"Failed to end session in store: {e}")

        # Remove session from memory
        del self._sessions[session_id]
        if admin_user_id in self._admin_sessions:
            self._admin_sessions[admin_user_id] = [
                sid for sid in self._admin_sessions[admin_user_id] if sid != session_id
            ]

        return True, f"Session ended ({session.actions_performed} actions performed)"

    def log_impersonation_action(
        self,
        session_id: str,
        action_type: str,
        action_details: Dict[str, Any],
        ip_address: str,
        user_agent: str,
    ) -> bool:
        """
        Log an action performed during impersonation.

        Args:
            session_id: Active session ID
            action_type: Type of action performed
            action_details: Details of the action
            ip_address: Request IP address
            user_agent: Request user agent

        Returns:
            True if logged successfully, False if session invalid
        """
        session = self._sessions.get(session_id)
        if not session:
            return False

        if session.is_expired():
            self._cleanup_expired_session(session_id)
            return False

        session.actions_performed += 1

        # Update action count in store
        store = self._get_store()
        if store:
            try:
                store.update_session_actions(session_id, session.actions_performed)
            except Exception as e:
                logger.error(f"Failed to update session actions in store: {e}")

        entry = ImpersonationAuditEntry(
            timestamp=datetime.now(timezone.utc),
            event_type="action",
            session_id=session_id,
            admin_user_id=session.admin_user_id,
            target_user_id=session.target_user_id,
            reason=session.reason,
            action_details={
                "action_type": action_type,
                "action_number": session.actions_performed,
                **action_details,
            },
            ip_address=ip_address,
            user_agent=user_agent,
            success=True,
        )
        self._log_audit(entry)
        return True

    def validate_session(self, session_id: str) -> Optional[ImpersonationSession]:
        """
        Validate and return an impersonation session.

        Args:
            session_id: Session to validate

        Returns:
            Session if valid, None if expired or not found
        """
        session = self._sessions.get(session_id)
        if not session:
            return None

        if session.is_expired():
            self._cleanup_expired_session(session_id)
            return None

        return session

    def _cleanup_expired_session(self, session_id: str) -> None:
        """Clean up an expired session with audit logging."""
        session = self._sessions.get(session_id)
        if not session:
            return

        entry = ImpersonationAuditEntry(
            timestamp=datetime.now(timezone.utc),
            event_type="timeout",
            session_id=session_id,
            admin_user_id=session.admin_user_id,
            target_user_id=session.target_user_id,
            reason=session.reason,
            action_details={"actions_performed": session.actions_performed},
            ip_address=session.ip_address,
            user_agent=session.user_agent,
            success=True,
        )
        self._log_audit(entry)

        # Mark session as ended in store
        store = self._get_store()
        if store:
            try:
                store.end_session(
                    session_id=session_id,
                    ended_by="timeout",
                    actions_performed=session.actions_performed,
                )
            except Exception as e:
                logger.error(f"Failed to end session in store: {e}")

        del self._sessions[session_id]
        admin_id = session.admin_user_id
        if admin_id in self._admin_sessions:
            self._admin_sessions[admin_id] = [
                sid for sid in self._admin_sessions[admin_id] if sid != session_id
            ]

    def get_active_sessions_for_admin(self, admin_user_id: str) -> List[ImpersonationSession]:
        """Get all active sessions for an admin."""
        session_ids = self._admin_sessions.get(admin_user_id, [])
        sessions = []
        for sid in session_ids:
            session = self.validate_session(sid)
            if session:
                sessions.append(session)
        return sessions

    def get_audit_log(
        self,
        admin_user_id: Optional[str] = None,
        target_user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[ImpersonationAuditEntry]:
        """
        Query the in-memory audit log.

        Args:
            admin_user_id: Filter by admin
            target_user_id: Filter by target
            event_type: Filter by event type
            since: Filter by timestamp
            limit: Maximum entries to return

        Returns:
            List of matching audit entries (newest first)
        """
        result = []
        for entry in reversed(self._audit_log):
            if admin_user_id and entry.admin_user_id != admin_user_id:
                continue
            if target_user_id and entry.target_user_id != target_user_id:
                continue
            if event_type and entry.event_type != event_type:
                continue
            if since and entry.timestamp < since:
                continue
            result.append(entry)
            if len(result) >= limit:
                break
        return result


# Global manager instance
_impersonation_manager: Optional[ImpersonationManager] = None


def get_impersonation_manager() -> ImpersonationManager:
    """Get or create the global impersonation manager."""
    global _impersonation_manager
    if _impersonation_manager is None:
        _impersonation_manager = ImpersonationManager()
    return _impersonation_manager


def configure_impersonation_manager(
    audit_callback: Optional[Callable[[ImpersonationAuditEntry], None]] = None,
    notification_callback: Optional[Callable[[str, str, str], None]] = None,
    require_2fa_for_admin_targets: bool = True,
    max_concurrent_sessions: int = 3,
) -> ImpersonationManager:
    """Configure the global impersonation manager."""
    global _impersonation_manager
    _impersonation_manager = ImpersonationManager(
        audit_callback=audit_callback,
        notification_callback=notification_callback,
        require_2fa_for_admin_targets=require_2fa_for_admin_targets,
        max_concurrent_sessions=max_concurrent_sessions,
    )
    return _impersonation_manager


def recover_impersonation_sessions() -> int:
    """
    Recover active impersonation sessions from the persistent store.

    Call this on server startup to ensure active sessions survive restarts.
    Returns the number of sessions recovered.

    This function is idempotent - calling it multiple times is safe.
    """
    global _sessions_recovered

    if _sessions_recovered:
        logger.debug("Impersonation sessions already recovered, skipping")
        return 0

    manager = get_impersonation_manager()
    store = manager._get_store()

    if not store:
        logger.debug("ImpersonationStore not available, cannot recover sessions")
        return 0

    recovered = 0
    try:
        active_sessions = store.get_active_sessions()

        for record in active_sessions:
            if record.session_id in manager._sessions:
                # Already in memory, skip
                continue

            # Reconstruct ImpersonationSession from stored record
            session = ImpersonationSession(
                session_id=record.session_id,
                admin_user_id=record.admin_user_id,
                admin_email=record.admin_email,
                target_user_id=record.target_user_id,
                target_email=record.target_email,
                reason=record.reason,
                started_at=record.started_at,
                expires_at=record.expires_at,
                ip_address=record.ip_address,
                user_agent=record.user_agent,
                actions_performed=record.actions_performed,
            )

            # Add to in-memory structures
            manager._sessions[record.session_id] = session
            if record.admin_user_id not in manager._admin_sessions:
                manager._admin_sessions[record.admin_user_id] = []
            manager._admin_sessions[record.admin_user_id].append(record.session_id)

            recovered += 1

        _sessions_recovered = True

        if recovered > 0:
            logger.info(f"Recovered {recovered} active impersonation sessions from store")

    except Exception as e:
        logger.warning(f"Failed to recover impersonation sessions: {e}")

    return recovered


def reset_session_recovery() -> None:
    """Reset recovery state (for testing)."""
    global _sessions_recovered
    _sessions_recovered = False


def clear_impersonation_sessions() -> int:
    """
    Clear all sessions from memory (for testing).

    Returns the number of sessions cleared.
    """
    manager = get_impersonation_manager()
    count = len(manager._sessions)
    manager._sessions.clear()
    manager._admin_sessions.clear()
    manager._audit_log.clear()
    return count
