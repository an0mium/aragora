"""
Aragora Audit Log System.

Enterprise-grade audit logging for compliance requirements:
- SOC 2 Type II
- HIPAA
- GDPR
- SOX

Features:
- Immutable audit trail with cryptographic integrity
- Retention policies (configurable)
- Export in compliance formats (JSON, CSV, SOC2)
- Full-text search
- Tamper detection via hash chains

Usage:
    from aragora.audit.log import AuditLog, AuditEvent, AuditCategory

    audit = AuditLog()

    # Log an event
    audit.log(AuditEvent(
        category=AuditCategory.AUTH,
        action="login",
        actor_id="user_123",
        resource_type="session",
        resource_id="sess_abc",
        outcome="success",
    ))

    # Export for compliance
    audit.export_soc2("soc2_audit_2026_Q1.json", start_date, end_date)
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Generator, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class AuditCategory(Enum):
    """Categories of audit events."""

    AUTH = "auth"  # Authentication events
    ACCESS = "access"  # Resource access
    DATA = "data"  # Data modifications
    ADMIN = "admin"  # Administrative actions
    BILLING = "billing"  # Billing/subscription events
    DEBATE = "debate"  # Debate operations
    API = "api"  # API access
    SECURITY = "security"  # Security events
    SYSTEM = "system"  # System events


class AuditOutcome(Enum):
    """Outcome of audited action."""

    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    ERROR = "error"


@dataclass
class AuditEvent:
    """An audit log event."""

    # Required fields
    category: AuditCategory
    action: str
    actor_id: str

    # Resource being acted upon
    resource_type: str = ""
    resource_id: str = ""

    # Outcome
    outcome: AuditOutcome = AuditOutcome.SUCCESS

    # Context
    ip_address: str = ""
    user_agent: str = ""
    correlation_id: str = ""
    org_id: str = ""
    workspace_id: str = ""

    # Details
    details: dict[str, Any] = field(default_factory=dict)
    reason: str = ""  # For denials/failures

    # Metadata
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Integrity
    previous_hash: str = ""
    event_hash: str = ""

    def compute_hash(self) -> str:
        """Compute hash of event for integrity chain."""
        data = (
            f"{self.id}|{self.timestamp.isoformat()}|{self.category.value}|"
            f"{self.action}|{self.actor_id}|{self.resource_type}|{self.resource_id}|"
            f"{self.outcome.value}|{self.previous_hash}"
        )
        return hashlib.sha256(data.encode()).hexdigest()[:32]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "action": self.action,
            "actor_id": self.actor_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "outcome": self.outcome.value,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "correlation_id": self.correlation_id,
            "org_id": self.org_id,
            "workspace_id": self.workspace_id,
            "details": self.details,
            "reason": self.reason,
            "previous_hash": self.previous_hash,
            "event_hash": self.event_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEvent":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if isinstance(data.get("timestamp"), str)
            else data.get("timestamp", datetime.utcnow()),
            category=AuditCategory(data.get("category", "system")),
            action=data.get("action", ""),
            actor_id=data.get("actor_id", ""),
            resource_type=data.get("resource_type", ""),
            resource_id=data.get("resource_id", ""),
            outcome=AuditOutcome(data.get("outcome", "success")),
            ip_address=data.get("ip_address", ""),
            user_agent=data.get("user_agent", ""),
            correlation_id=data.get("correlation_id", ""),
            org_id=data.get("org_id", ""),
            workspace_id=data.get("workspace_id", ""),
            details=data.get("details", {}),
            reason=data.get("reason", ""),
            previous_hash=data.get("previous_hash", ""),
            event_hash=data.get("event_hash", ""),
        )


@dataclass
class AuditQuery:
    """Query parameters for audit log search."""

    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    category: Optional[AuditCategory] = None
    action: Optional[str] = None
    actor_id: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    outcome: Optional[AuditOutcome] = None
    org_id: Optional[str] = None
    ip_address: Optional[str] = None
    search_text: Optional[str] = None
    limit: int = 1000
    offset: int = 0


class AuditLog:
    """
    Enterprise audit log with compliance export support.

    Provides:
    - Immutable audit trail with hash chain integrity
    - Configurable retention policies
    - Export in compliance formats
    - Full-text search
    - Tamper detection
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        retention_days: int = 365 * 7,  # 7 years default (SOX requirement)
    ):
        """
        Initialize audit log.

        Args:
            db_path: Path to SQLite database (default: .nomic/audit.db)
            retention_days: Days to retain logs (default: 7 years for SOX)
        """
        if db_path is None:
            db_path = Path(".nomic/audit.db")
        self.db_path = db_path
        self.retention_days = retention_days
        self._last_hash = ""
        self._ensure_schema()
        self._load_last_hash()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        """Create database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    category TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    resource_type TEXT,
                    resource_id TEXT,
                    outcome TEXT NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    correlation_id TEXT,
                    org_id TEXT,
                    workspace_id TEXT,
                    details TEXT DEFAULT '{}',
                    reason TEXT,
                    previous_hash TEXT,
                    event_hash TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_audit_category ON audit_events(category);
                CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_events(actor_id);
                CREATE INDEX IF NOT EXISTS idx_audit_org ON audit_events(org_id);
                CREATE INDEX IF NOT EXISTS idx_audit_resource
                    ON audit_events(resource_type, resource_id);
                CREATE INDEX IF NOT EXISTS idx_audit_outcome ON audit_events(outcome);

                -- Full-text search
                CREATE VIRTUAL TABLE IF NOT EXISTS audit_fts USING fts5(
                    id, action, actor_id, resource_type, resource_id, details, reason,
                    content='audit_events',
                    content_rowid='rowid'
                );

                -- Triggers for FTS sync
                CREATE TRIGGER IF NOT EXISTS audit_ai AFTER INSERT ON audit_events BEGIN
                    INSERT INTO audit_fts(rowid, id, action, actor_id, resource_type,
                        resource_id, details, reason)
                    VALUES (new.rowid, new.id, new.action, new.actor_id, new.resource_type,
                        new.resource_id, new.details, new.reason);
                END;
            """
            )
            conn.commit()

    def _load_last_hash(self) -> None:
        """Load the last event hash for chain continuity."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT event_hash FROM audit_events ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if row:
                self._last_hash = row["event_hash"]

    def log(self, event: AuditEvent) -> str:
        """
        Log an audit event.

        Args:
            event: The audit event to log

        Returns:
            Event ID
        """
        # Set hash chain
        event.previous_hash = self._last_hash
        event.event_hash = event.compute_hash()

        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_events
                (id, timestamp, category, action, actor_id, resource_type, resource_id,
                 outcome, ip_address, user_agent, correlation_id, org_id, workspace_id,
                 details, reason, previous_hash, event_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.timestamp.isoformat(),
                    event.category.value,
                    event.action,
                    event.actor_id,
                    event.resource_type,
                    event.resource_id,
                    event.outcome.value,
                    event.ip_address,
                    event.user_agent,
                    event.correlation_id,
                    event.org_id,
                    event.workspace_id,
                    json.dumps(event.details),
                    event.reason,
                    event.previous_hash,
                    event.event_hash,
                ),
            )
            conn.commit()

        self._last_hash = event.event_hash

        logger.debug(
            f"audit_logged category={event.category.value} action={event.action} "
            f"actor={event.actor_id} outcome={event.outcome.value}"
        )

        return event.id

    def query(self, query: AuditQuery) -> list[AuditEvent]:
        """
        Query audit events.

        Args:
            query: Query parameters

        Returns:
            List of matching events
        """
        conditions = []
        params: list[Any] = []

        if query.start_date:
            conditions.append("timestamp >= ?")
            params.append(query.start_date.isoformat())

        if query.end_date:
            conditions.append("timestamp <= ?")
            params.append(query.end_date.isoformat())

        if query.category:
            conditions.append("category = ?")
            params.append(query.category.value)

        if query.action:
            conditions.append("action = ?")
            params.append(query.action)

        if query.actor_id:
            conditions.append("actor_id = ?")
            params.append(query.actor_id)

        if query.resource_type:
            conditions.append("resource_type = ?")
            params.append(query.resource_type)

        if query.resource_id:
            conditions.append("resource_id = ?")
            params.append(query.resource_id)

        if query.outcome:
            conditions.append("outcome = ?")
            params.append(query.outcome.value)

        if query.org_id:
            conditions.append("org_id = ?")
            params.append(query.org_id)

        if query.ip_address:
            conditions.append("ip_address = ?")
            params.append(query.ip_address)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Handle full-text search separately
        if query.search_text:
            sql = f"""
                SELECT e.* FROM audit_events e
                JOIN audit_fts f ON e.id = f.id
                WHERE f.audit_fts MATCH ? AND {where_clause}
                ORDER BY e.timestamp DESC
                LIMIT ? OFFSET ?
            """
            params = [query.search_text] + params + [query.limit, query.offset]
        else:
            sql = f"""
                SELECT * FROM audit_events
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """
            params.extend([query.limit, query.offset])

        events = []
        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            for row in rows:
                event_data = dict(row)
                event_data["details"] = json.loads(event_data.get("details", "{}"))
                events.append(AuditEvent.from_dict(event_data))

        return events

    def verify_integrity(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> tuple[bool, list[str]]:
        """
        Verify audit log integrity via hash chain.

        Args:
            start_date: Start of verification range
            end_date: End of verification range

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        conditions = []
        params: list[Any] = []

        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date.isoformat())
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date.isoformat())

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM audit_events WHERE {where_clause} ORDER BY timestamp",
                params,
            ).fetchall()

            prev_hash = ""
            for row in rows:
                event_data = dict(row)
                event_data["details"] = json.loads(event_data.get("details", "{}"))
                event = AuditEvent.from_dict(event_data)

                # Verify previous hash
                if event.previous_hash != prev_hash:
                    errors.append(
                        f"Hash chain broken at event {event.id}: "
                        f"expected previous_hash={prev_hash}, got {event.previous_hash}"
                    )

                # Verify event hash
                computed = event.compute_hash()
                if event.event_hash != computed:
                    errors.append(
                        f"Event {event.id} hash mismatch: "
                        f"stored={event.event_hash}, computed={computed}"
                    )

                prev_hash = event.event_hash

        return len(errors) == 0, errors

    def export_json(
        self,
        output_path: Path,
        start_date: datetime,
        end_date: datetime,
        org_id: Optional[str] = None,
    ) -> int:
        """
        Export audit log to JSON.

        Args:
            output_path: Output file path
            start_date: Export start date
            end_date: Export end date
            org_id: Filter by organization

        Returns:
            Number of events exported
        """
        query = AuditQuery(
            start_date=start_date,
            end_date=end_date,
            org_id=org_id,
            limit=100000,
        )
        events = self.query(query)

        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "org_id": org_id,
            "event_count": len(events),
            "events": [e.to_dict() for e in events],
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        return len(events)

    def export_csv(
        self,
        output_path: Path,
        start_date: datetime,
        end_date: datetime,
        org_id: Optional[str] = None,
    ) -> int:
        """
        Export audit log to CSV.

        Args:
            output_path: Output file path
            start_date: Export start date
            end_date: Export end date
            org_id: Filter by organization

        Returns:
            Number of events exported
        """
        query = AuditQuery(
            start_date=start_date,
            end_date=end_date,
            org_id=org_id,
            limit=100000,
        )
        events = self.query(query)

        fieldnames = [
            "id",
            "timestamp",
            "category",
            "action",
            "actor_id",
            "resource_type",
            "resource_id",
            "outcome",
            "ip_address",
            "org_id",
            "reason",
            "event_hash",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for event in events:
                row = event.to_dict()
                writer.writerow({k: row.get(k, "") for k in fieldnames})

        return len(events)

    def export_soc2(
        self,
        output_path: Path,
        start_date: datetime,
        end_date: datetime,
        org_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Export audit log in SOC 2 Type II format.

        Generates a comprehensive audit report suitable for SOC 2 auditors.

        Args:
            output_path: Output file path
            start_date: Audit period start
            end_date: Audit period end
            org_id: Organization to audit

        Returns:
            Summary statistics
        """
        query = AuditQuery(
            start_date=start_date,
            end_date=end_date,
            org_id=org_id,
            limit=100000,
        )
        events = self.query(query)

        # Verify integrity
        is_valid, integrity_errors = self.verify_integrity(start_date, end_date)

        # Compute statistics with proper types
        total_events = len(events)
        by_category: dict[str, int] = {}
        by_outcome: dict[str, int] = {}
        by_action: dict[str, int] = {}
        unique_actors: set[str] = set()
        auth_failures = 0
        access_denials = 0
        security_events = 0

        for event in events:
            # Category counts
            cat = event.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

            # Outcome counts
            out = event.outcome.value
            by_outcome[out] = by_outcome.get(out, 0) + 1

            # Action counts
            act = f"{cat}:{event.action}"
            by_action[act] = by_action.get(act, 0) + 1

            # Unique actors
            unique_actors.add(event.actor_id)

            # Special counters
            if event.category == AuditCategory.AUTH and event.outcome == AuditOutcome.FAILURE:
                auth_failures += 1
            if event.outcome == AuditOutcome.DENIED:
                access_denials += 1
            if event.category == AuditCategory.SECURITY:
                security_events += 1

        # Build SOC 2 report
        report = {
            "report_type": "SOC 2 Type II Audit Log Export",
            "generated_at": datetime.utcnow().isoformat(),
            "audit_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": (end_date - start_date).days,
            },
            "organization": org_id or "all",
            "integrity": {
                "chain_verified": is_valid,
                "errors": integrity_errors[:10] if integrity_errors else [],
                "total_errors": len(integrity_errors),
            },
            "summary": {
                "total_events": total_events,
                "unique_actors": len(unique_actors),
                "categories": by_category,
                "outcomes": by_outcome,
            },
            "security_metrics": {
                "authentication_failures": auth_failures,
                "access_denials": access_denials,
                "security_events": security_events,
            },
            "control_evidence": {
                "CC6.1_logical_access": {
                    "login_events": by_category.get("auth", 0),
                    "access_events": by_category.get("access", 0),
                },
                "CC6.2_access_removal": {
                    "relevant_actions": by_action.get("auth:logout", 0)
                    + by_action.get("admin:revoke_access", 0),
                },
                "CC6.3_access_authorization": {
                    "denied_attempts": access_denials,
                },
                "CC7.2_security_events": {
                    "total_security_events": security_events,
                },
            },
            "events": [e.to_dict() for e in events],
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        return {
            "events_exported": len(events),
            "integrity_verified": is_valid,
            "integrity_errors": len(integrity_errors),
        }

    def apply_retention(self) -> int:
        """
        Apply retention policy and delete old events.

        Returns:
            Number of events deleted
        """
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)

        with self._connection() as conn:
            result = conn.execute(
                "DELETE FROM audit_events WHERE timestamp < ?",
                (cutoff.isoformat(),),
            )
            deleted = result.rowcount
            conn.commit()

        if deleted > 0:
            logger.info(f"audit_retention_applied deleted={deleted} cutoff={cutoff.date()}")

        return deleted

    def get_stats(self) -> dict[str, Any]:
        """Get audit log statistics."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    MIN(timestamp) as oldest,
                    MAX(timestamp) as newest
                FROM audit_events
                """
            ).fetchone()

            by_category = {}
            for cat_row in conn.execute(
                "SELECT category, COUNT(*) as count FROM audit_events GROUP BY category"
            ).fetchall():
                by_category[cat_row["category"]] = cat_row["count"]

            return {
                "total_events": row["total"] if row else 0,
                "oldest_event": row["oldest"] if row else None,
                "newest_event": row["newest"] if row else None,
                "by_category": by_category,
                "retention_days": self.retention_days,
            }


# Convenience functions for common audit events
def audit_auth_login(
    audit: AuditLog,
    user_id: str,
    ip_address: str = "",
    success: bool = True,
    reason: str = "",
) -> str:
    """Log authentication attempt."""
    return audit.log(
        AuditEvent(
            category=AuditCategory.AUTH,
            action="login",
            actor_id=user_id,
            resource_type="session",
            outcome=AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE,
            ip_address=ip_address,
            reason=reason,
        )
    )


def audit_data_access(
    audit: AuditLog,
    user_id: str,
    resource_type: str,
    resource_id: str,
    action: str = "read",
    org_id: str = "",
) -> str:
    """Log data access."""
    return audit.log(
        AuditEvent(
            category=AuditCategory.ACCESS,
            action=action,
            actor_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            org_id=org_id,
        )
    )


def audit_admin_action(
    audit: AuditLog,
    admin_id: str,
    action: str,
    target_type: str,
    target_id: str,
    details: Optional[dict] = None,
) -> str:
    """Log administrative action."""
    return audit.log(
        AuditEvent(
            category=AuditCategory.ADMIN,
            action=action,
            actor_id=admin_id,
            resource_type=target_type,
            resource_id=target_id,
            details=details or {},
        )
    )


__all__ = [
    "AuditCategory",
    "AuditEvent",
    "AuditLog",
    "AuditOutcome",
    "AuditQuery",
    "audit_admin_action",
    "audit_auth_login",
    "audit_data_access",
]
