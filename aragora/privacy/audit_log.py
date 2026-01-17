"""
Privacy Audit Log.

SOC2-compliant audit logging for privacy-sensitive operations.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class AuditAction(str, Enum):
    """Actions that can be audited."""

    # Data access
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXPORT = "export"

    # Workspace management
    CREATE_WORKSPACE = "create_workspace"
    DELETE_WORKSPACE = "delete_workspace"
    ADD_MEMBER = "add_member"
    REMOVE_MEMBER = "remove_member"
    MODIFY_PERMISSIONS = "modify_permissions"

    # Document operations
    UPLOAD_DOCUMENT = "upload_document"
    DELETE_DOCUMENT = "delete_document"
    CLASSIFY_DOCUMENT = "classify_document"

    # Query operations
    QUERY = "query"
    SEARCH = "search"

    # Administrative
    MODIFY_POLICY = "modify_policy"
    EXECUTE_RETENTION = "execute_retention"
    GENERATE_REPORT = "generate_report"

    # Authentication
    LOGIN = "login"
    LOGOUT = "logout"
    AUTH_FAILURE = "auth_failure"


class AuditOutcome(str, Enum):
    """Outcome of an audited action."""

    SUCCESS = "success"
    DENIED = "denied"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class Actor:
    """Entity performing an action."""

    id: str
    type: str = "user"  # user, system, agent
    name: str = ""
    ip_address: str = ""
    user_agent: str = ""
    session_id: str = ""


@dataclass
class Resource:
    """Resource being accessed."""

    id: str
    type: str  # workspace, document, fact, query
    workspace_id: str = ""
    name: str = ""
    sensitivity_level: str = ""


@dataclass
class AuditEntry:
    """A single audit log entry."""

    id: str
    timestamp: datetime
    action: AuditAction
    outcome: AuditOutcome
    actor: Actor
    resource: Resource

    # Details
    details: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0
    error_message: str = ""

    # Integrity
    checksum: str = ""
    previous_checksum: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "outcome": self.outcome.value,
            "actor": {
                "id": self.actor.id,
                "type": self.actor.type,
                "name": self.actor.name,
                "ip_address": self.actor.ip_address,
            },
            "resource": {
                "id": self.resource.id,
                "type": self.resource.type,
                "workspace_id": self.resource.workspace_id,
                "sensitivity_level": self.resource.sensitivity_level,
            },
            "details": self.details,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
            "checksum": self.checksum,
            "previous_checksum": self.previous_checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEntry":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            action=AuditAction(data["action"]),
            outcome=AuditOutcome(data["outcome"]),
            actor=Actor(
                id=data["actor"]["id"],
                type=data["actor"].get("type", "user"),
                name=data["actor"].get("name", ""),
                ip_address=data["actor"].get("ip_address", ""),
            ),
            resource=Resource(
                id=data["resource"]["id"],
                type=data["resource"]["type"],
                workspace_id=data["resource"].get("workspace_id", ""),
                sensitivity_level=data["resource"].get("sensitivity_level", ""),
            ),
            details=data.get("details", {}),
            duration_ms=data.get("duration_ms", 0),
            error_message=data.get("error_message", ""),
            checksum=data.get("checksum", ""),
            previous_checksum=data.get("previous_checksum", ""),
        )


@dataclass
class AuditLogConfig:
    """Configuration for audit logging."""

    # Storage
    log_directory: str = "/var/log/aragora/audit"
    max_file_size_mb: int = 100
    retention_days: int = 365 * 7  # 7 years for compliance

    # Behavior
    log_read_operations: bool = True
    log_query_operations: bool = True
    include_request_details: bool = True

    # Integrity
    enable_checksums: bool = True
    enable_chain_verification: bool = True

    # Filtering
    exclude_actors: list[str] = field(default_factory=list)
    exclude_resources: list[str] = field(default_factory=list)


class PrivacyAuditLog:
    """
    SOC2-compliant audit logging for privacy operations.

    Features:
    - Immutable append-only log
    - Cryptographic integrity verification (hash chain)
    - Configurable retention
    - Compliance report generation
    """

    def __init__(self, config: AuditLogConfig | None = None):
        self.config = config or AuditLogConfig()
        self._entries: list[AuditEntry] = []
        self._last_checksum: str = ""
        self._log_file: Path | None = None

        self._ensure_log_directory()

    def _ensure_log_directory(self) -> None:
        """Ensure log directory exists."""
        log_path = Path(self.config.log_directory)
        log_path.mkdir(parents=True, exist_ok=True)

    def _compute_checksum(self, entry: AuditEntry) -> str:
        """Compute checksum for an entry."""
        if not self.config.enable_checksums:
            return ""

        data = {
            "id": entry.id,
            "timestamp": entry.timestamp.isoformat(),
            "action": entry.action.value,
            "outcome": entry.outcome.value,
            "actor_id": entry.actor.id,
            "resource_id": entry.resource.id,
            "previous_checksum": entry.previous_checksum,
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    async def log(
        self,
        action: AuditAction,
        actor: Actor,
        resource: Resource,
        outcome: AuditOutcome,
        details: dict[str, Any] | None = None,
        duration_ms: int = 0,
        error_message: str = "",
    ) -> AuditEntry:
        """
        Log an audit entry.

        Args:
            action: The action being performed
            actor: Who is performing the action
            resource: What resource is being accessed
            outcome: Result of the action
            details: Additional context
            duration_ms: How long the operation took
            error_message: Error message if failed

        Returns:
            The created audit entry
        """
        # Check exclusions
        if actor.id in self.config.exclude_actors:
            return None
        if resource.id in self.config.exclude_resources:
            return None

        # Skip read/query if not logging
        if action == AuditAction.READ and not self.config.log_read_operations:
            return None
        if (
            action in (AuditAction.QUERY, AuditAction.SEARCH)
            and not self.config.log_query_operations
        ):
            return None

        entry = AuditEntry(
            id=f"audit_{uuid4().hex[:12]}",
            timestamp=datetime.utcnow(),
            action=action,
            outcome=outcome,
            actor=actor,
            resource=resource,
            details=details or {},
            duration_ms=duration_ms,
            error_message=error_message,
            previous_checksum=self._last_checksum,
        )

        # Compute checksum
        entry.checksum = self._compute_checksum(entry)
        self._last_checksum = entry.checksum

        # Store entry
        self._entries.append(entry)

        # Write to file
        await self._write_entry(entry)

        logger.debug(
            f"Audit: {action.value} by {actor.id} on {resource.type}/{resource.id} -> {outcome.value}"
        )

        return entry

    async def _write_entry(self, entry: AuditEntry) -> None:
        """Write entry to log file."""
        log_path = Path(self.config.log_directory)
        date_str = entry.timestamp.strftime("%Y-%m-%d")
        log_file = log_path / f"audit_{date_str}.jsonl"

        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except OSError as e:
            logger.error(f"Failed to write audit log: {e}")

    async def query(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        actor_id: str | None = None,
        resource_id: str | None = None,
        workspace_id: str | None = None,
        action: AuditAction | None = None,
        outcome: AuditOutcome | None = None,
        limit: int = 1000,
    ) -> list[AuditEntry]:
        """
        Query audit log entries.

        Args:
            start_date: Filter by start date
            end_date: Filter by end date
            actor_id: Filter by actor
            resource_id: Filter by resource
            workspace_id: Filter by workspace
            action: Filter by action type
            outcome: Filter by outcome
            limit: Maximum entries to return

        Returns:
            Matching audit entries
        """
        entries = self._entries

        if start_date:
            entries = [e for e in entries if e.timestamp >= start_date]
        if end_date:
            entries = [e for e in entries if e.timestamp <= end_date]
        if actor_id:
            entries = [e for e in entries if e.actor.id == actor_id]
        if resource_id:
            entries = [e for e in entries if e.resource.id == resource_id]
        if workspace_id:
            entries = [e for e in entries if e.resource.workspace_id == workspace_id]
        if action:
            entries = [e for e in entries if e.action == action]
        if outcome:
            entries = [e for e in entries if e.outcome == outcome]

        return entries[:limit]

    async def verify_integrity(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Verify the integrity of the audit log chain.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if not self.config.enable_chain_verification:
            return True, []

        entries = await self.query(start_date=start_date, end_date=end_date, limit=100000)
        errors = []

        previous_checksum = ""
        for i, entry in enumerate(entries):
            # Verify previous checksum reference
            if entry.previous_checksum != previous_checksum:
                errors.append(f"Entry {entry.id}: previous_checksum mismatch at position {i}")

            # Verify checksum computation
            expected_checksum = self._compute_checksum(entry)
            if entry.checksum != expected_checksum:
                errors.append(
                    f"Entry {entry.id}: checksum mismatch (expected {expected_checksum[:8]}...)"
                )

            previous_checksum = entry.checksum

        return len(errors) == 0, errors

    async def generate_compliance_report(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        workspace_id: str | None = None,
        format: str = "json",
    ) -> dict[str, Any]:
        """
        Generate a compliance report for the specified period.

        Args:
            start_date: Report start date
            end_date: Report end date
            workspace_id: Filter by workspace
            format: Output format (json, summary)

        Returns:
            Compliance report data
        """
        start_date = start_date or (datetime.utcnow() - timedelta(days=30))
        end_date = end_date or datetime.utcnow()

        entries = await self.query(
            start_date=start_date,
            end_date=end_date,
            workspace_id=workspace_id,
            limit=100000,
        )

        # Aggregate statistics
        by_action: dict[str, int] = {}
        by_outcome: dict[str, int] = {}
        by_actor: dict[str, int] = {}
        by_resource_type: dict[str, int] = {}
        denied_count = 0
        failed_count = 0

        for entry in entries:
            by_action[entry.action.value] = by_action.get(entry.action.value, 0) + 1
            by_outcome[entry.outcome.value] = by_outcome.get(entry.outcome.value, 0) + 1
            by_actor[entry.actor.id] = by_actor.get(entry.actor.id, 0) + 1
            by_resource_type[entry.resource.type] = by_resource_type.get(entry.resource.type, 0) + 1

            if entry.outcome == AuditOutcome.DENIED:
                denied_count += 1
            elif entry.outcome == AuditOutcome.FAILED:
                failed_count += 1

        # Verify integrity
        is_valid, integrity_errors = await self.verify_integrity(start_date, end_date)

        report = {
            "report_id": f"compliance_{uuid4().hex[:8]}",
            "generated_at": datetime.utcnow().isoformat(),
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "workspace_id": workspace_id,
            "summary": {
                "total_entries": len(entries),
                "denied_count": denied_count,
                "failed_count": failed_count,
                "unique_actors": len(by_actor),
            },
            "by_action": by_action,
            "by_outcome": by_outcome,
            "by_resource_type": by_resource_type,
            "top_actors": sorted(by_actor.items(), key=lambda x: x[1], reverse=True)[:10],
            "integrity": {
                "verified": is_valid,
                "errors": integrity_errors[:10] if integrity_errors else [],
            },
        }

        return report

    async def get_actor_history(
        self,
        actor_id: str,
        days: int = 30,
    ) -> list[AuditEntry]:
        """Get all actions by a specific actor."""
        start_date = datetime.utcnow() - timedelta(days=days)
        return await self.query(actor_id=actor_id, start_date=start_date)

    async def get_resource_history(
        self,
        resource_id: str,
        days: int = 30,
    ) -> list[AuditEntry]:
        """Get all actions on a specific resource."""
        start_date = datetime.utcnow() - timedelta(days=days)
        return await self.query(resource_id=resource_id, start_date=start_date)

    async def get_denied_access_attempts(
        self,
        days: int = 7,
    ) -> list[AuditEntry]:
        """Get all denied access attempts."""
        start_date = datetime.utcnow() - timedelta(days=days)
        return await self.query(
            start_date=start_date,
            outcome=AuditOutcome.DENIED,
        )

    async def export_logs(
        self,
        start_date: datetime,
        end_date: datetime,
        output_path: Path,
    ) -> int:
        """
        Export audit logs to a file for archival.

        Returns:
            Number of entries exported
        """
        entries = await self.query(start_date=start_date, end_date=end_date, limit=1000000)

        with open(output_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry.to_dict()) + "\n")

        logger.info(f"Exported {len(entries)} audit entries to {output_path}")
        return len(entries)

    async def cleanup_old_entries(
        self,
        retention_days: int | None = None,
    ) -> int:
        """
        Remove entries older than retention period.

        Returns:
            Number of entries removed
        """
        retention = retention_days or self.config.retention_days
        cutoff = datetime.utcnow() - timedelta(days=retention)

        original_count = len(self._entries)
        self._entries = [e for e in self._entries if e.timestamp >= cutoff]
        removed_count = original_count - len(self._entries)

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} audit entries older than {retention} days")

        return removed_count


# Global instance
_audit_log: PrivacyAuditLog | None = None


def get_audit_log(config: AuditLogConfig | None = None) -> PrivacyAuditLog:
    """Get or create the global audit log."""
    global _audit_log
    if _audit_log is None:
        _audit_log = PrivacyAuditLog(config)
    return _audit_log


__all__ = [
    "PrivacyAuditLog",
    "AuditEntry",
    "AuditAction",
    "AuditOutcome",
    "AuditLogConfig",
    "Actor",
    "Resource",
    "get_audit_log",
]
