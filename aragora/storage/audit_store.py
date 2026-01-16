"""
AuditStore - SQLite backend for audit logging and compliance.

DEPRECATION NOTICE:
    This module provides user/organization-specific audit storage. For new code,
    use the unified audit system in aragora.audit instead:

        from aragora.audit import AuditLog, AuditEvent, AuditCategory

    The unified system supports all audit categories including AUTH, DATA,
    ADMIN, and BILLING events with compliance export (SOC 2, GDPR, HIPAA).

    This module is maintained for backward compatibility with existing
    storage code that depends on AuditStore for UserStore composition.

Extracted from UserStore to improve modularity.
Provides audit trail functionality for:
- User actions (login, logout, settings changes)
- Organization changes (member added/removed, tier changes)
- Subscription events (created, updated, cancelled)
- Security events (failed logins, password changes)
"""

from __future__ import annotations

__all__ = [
    "AuditStore",
]

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger(__name__)


class AuditStore:
    """
    SQLite-backed storage for audit logging.

    Can be used standalone or composed with UserStore.
    Thread-safe with connection pooling via thread-local storage.
    """

    def __init__(
        self,
        db_path: Path | str,
        get_connection: Optional[Callable[[], sqlite3.Connection]] = None,
    ):
        """
        Initialize AuditStore.

        Args:
            db_path: Path to SQLite database file
            get_connection: Optional connection factory (for sharing with UserStore)
        """
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._external_get_connection = get_connection

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if self._external_get_connection:
            return self._external_get_connection()

        if not hasattr(self._local, "connection"):
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.connection = conn
        return self._local.connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        """Context manager for database transactions."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.debug("Transaction rolled back due to: %s", e)
            raise

    # =========================================================================
    # Audit Logging Methods
    # =========================================================================

    def log_event(
        self,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        old_value: Optional[dict] = None,
        new_value: Optional[dict] = None,
        metadata: Optional[dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> int:
        """
        Log an audit event.

        Args:
            action: Action performed (e.g., 'subscription.created', 'tier.changed')
            resource_type: Type of resource (e.g., 'subscription', 'user', 'organization')
            resource_id: ID of the affected resource
            user_id: User who performed the action
            org_id: Organization context
            old_value: Previous value (for changes)
            new_value: New value (for changes)
            metadata: Additional context
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Audit log entry ID
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO audit_log
                (timestamp, user_id, org_id, action, resource_type, resource_id,
                 old_value, new_value, metadata, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    user_id,
                    org_id,
                    action,
                    resource_type,
                    resource_id,
                    json.dumps(old_value) if old_value else None,
                    json.dumps(new_value) if new_value else None,
                    json.dumps(metadata or {}),
                    ip_address,
                    user_agent,
                ),
            )
            return cursor.lastrowid or 0

    def get_log(
        self,
        org_id: Optional[str] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """
        Query audit log entries.

        Args:
            org_id: Filter by organization
            user_id: Filter by user
            action: Filter by action (supports prefix match with *)
            resource_type: Filter by resource type
            since: Filter entries after this time
            until: Filter entries before this time
            limit: Maximum entries to return
            offset: Pagination offset

        Returns:
            List of audit log entries
        """
        conditions: list[str] = []
        params: list[Any] = []

        if org_id:
            conditions.append("org_id = ?")
            params.append(org_id)
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if action:
            if action.endswith("*"):
                conditions.append("action LIKE ?")
                params.append(action[:-1] + "%")
            else:
                conditions.append("action = ?")
                params.append(action)
        if resource_type:
            conditions.append("resource_type = ?")
            params.append(resource_type)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())
        if until:
            conditions.append("timestamp <= ?")
            params.append(until.isoformat())

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.extend([limit, offset])

        with self._transaction() as cursor:
            query = f"""
                SELECT id, timestamp, user_id, org_id, action, resource_type,
                       resource_id, old_value, new_value, metadata, ip_address, user_agent
                FROM audit_log
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """  # nosec B608 - where_clause built from hardcoded conditions
            cursor.execute(query, params)
            return [
                {
                    "id": row[0],
                    "timestamp": row[1],
                    "user_id": row[2],
                    "org_id": row[3],
                    "action": row[4],
                    "resource_type": row[5],
                    "resource_id": row[6],
                    "old_value": json.loads(row[7]) if row[7] else None,
                    "new_value": json.loads(row[8]) if row[8] else None,
                    "metadata": json.loads(row[9]) if row[9] else {},
                    "ip_address": row[10],
                    "user_agent": row[11],
                }
                for row in cursor.fetchall()
            ]

    def get_log_count(
        self,
        org_id: Optional[str] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
    ) -> int:
        """Get count of audit log entries matching filters."""
        conditions: list[str] = []
        params: list[Any] = []

        if org_id:
            conditions.append("org_id = ?")
            params.append(org_id)
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if action:
            if action.endswith("*"):
                conditions.append("action LIKE ?")
                params.append(action[:-1] + "%")
            else:
                conditions.append("action = ?")
                params.append(action)
        if resource_type:
            conditions.append("resource_type = ?")
            params.append(resource_type)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._transaction() as cursor:
            query = f"SELECT COUNT(*) FROM audit_log WHERE {where_clause}"  # nosec B608
            cursor.execute(query, params)
            return cursor.fetchone()[0]

    def cleanup_old_entries(self, days: int = 90) -> int:
        """
        Delete audit log entries older than specified days.

        Args:
            days: Number of days to retain (default 90)

        Returns:
            Number of deleted entries
        """
        cutoff = datetime.utcnow().isoformat()
        # Calculate cutoff date
        from datetime import timedelta

        cutoff_date = datetime.utcnow() - timedelta(days=days)
        cutoff = cutoff_date.isoformat()

        with self._transaction() as cursor:
            cursor.execute(
                "DELETE FROM audit_log WHERE timestamp < ?",
                (cutoff,),
            )
            count = cursor.rowcount
            if count > 0:
                logger.info(f"Cleaned up {count} audit log entries older than {days} days")
            return count

    def get_recent_activity(
        self,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        hours: int = 24,
        limit: int = 50,
    ) -> list[dict]:
        """
        Get recent activity for a user or organization.

        Args:
            user_id: Filter by user
            org_id: Filter by organization
            hours: Look back period in hours
            limit: Maximum entries to return

        Returns:
            List of recent audit events
        """
        from datetime import timedelta

        since = datetime.utcnow() - timedelta(hours=hours)
        return self.get_log(
            user_id=user_id,
            org_id=org_id,
            since=since,
            limit=limit,
        )

    def get_security_events(
        self,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Get security-related audit events.

        Includes: login attempts, password changes, API key operations,
        permission changes, etc.

        Args:
            user_id: Filter by user
            org_id: Filter by organization
            limit: Maximum entries to return

        Returns:
            List of security audit events
        """
        # Security-related action prefixes
        security_actions = [
            "login.%",
            "password.%",
            "api_key.%",
            "permission.%",
            "auth.%",
            "lockout.%",
        ]

        all_events = []
        for action_pattern in security_actions:
            events = self.get_log(
                user_id=user_id,
                org_id=org_id,
                action=action_pattern.replace("%", "*"),
                limit=limit,
            )
            all_events.extend(events)

        # Sort by timestamp descending and limit
        all_events.sort(key=lambda e: e["timestamp"], reverse=True)
        return all_events[:limit]

    def close(self) -> None:
        """Close database connection if we own it."""
        if self._external_get_connection is None and hasattr(self._local, "connection"):
            self._local.connection.close()
            del self._local.connection


# Backwards compatibility alias
log_audit_event = AuditStore.log_event
get_audit_log = AuditStore.get_log
get_audit_log_count = AuditStore.get_log_count
