"""
Immutable Audit Log for the Aragora Control Plane.

Provides compliance-ready audit trail for all control plane operations:
- Append-only log storage using Redis Streams
- Tamper-evident entries with cryptographic hashing
- Query API for filtering by actor, action, date range
- Export capabilities for compliance reporting

The audit log captures:
- Agent registrations/deregistrations
- Task lifecycle events
- Deliberation outcomes
- Policy evaluations
- Configuration changes
- Authentication events
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Enums
# =============================================================================

AUDIT_STREAM_KEY = "aragora:audit:log"
AUDIT_RETENTION_DAYS = 90  # Default retention period


class AuditAction(Enum):
    """Types of auditable actions."""

    # Agent actions
    AGENT_REGISTERED = "agent.registered"
    AGENT_UNREGISTERED = "agent.unregistered"
    AGENT_STATUS_CHANGED = "agent.status_changed"
    AGENT_CONFIG_UPDATED = "agent.config_updated"

    # Task actions
    TASK_SUBMITTED = "task.submitted"
    TASK_CLAIMED = "task.claimed"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_CANCELLED = "task.cancelled"
    TASK_RETRIED = "task.retried"

    # Deliberation actions
    DELIBERATION_STARTED = "deliberation.started"
    DELIBERATION_ROUND_COMPLETED = "deliberation.round_completed"
    DELIBERATION_CONSENSUS = "deliberation.consensus"
    DELIBERATION_FAILED = "deliberation.failed"
    DELIBERATION_TIMEOUT = "deliberation.timeout"

    # Policy actions
    POLICY_EVALUATED = "policy.evaluated"
    POLICY_VIOLATION = "policy.violation"
    POLICY_UPDATED = "policy.updated"
    POLICY_DECISION_ALLOW = "policy.decision_allow"
    POLICY_DECISION_DENY = "policy.decision_deny"
    POLICY_DECISION_WARN = "policy.decision_warn"

    # Deliberation SLA actions
    DELIBERATION_SLA_WARNING = "deliberation.sla_warning"
    DELIBERATION_SLA_CRITICAL = "deliberation.sla_critical"
    DELIBERATION_SLA_VIOLATED = "deliberation.sla_violated"

    # Authentication actions
    AUTH_LOGIN = "auth.login"
    AUTH_LOGOUT = "auth.logout"
    AUTH_TOKEN_ISSUED = "auth.token_issued"
    AUTH_TOKEN_REVOKED = "auth.token_revoked"

    # Configuration actions
    CONFIG_UPDATED = "config.updated"
    WORKSPACE_CREATED = "workspace.created"
    WORKSPACE_DELETED = "workspace.deleted"
    CONNECTOR_ADDED = "connector.added"
    CONNECTOR_REMOVED = "connector.removed"

    # System actions
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"


class ActorType(Enum):
    """Types of actors that can perform actions."""

    AGENT = "agent"
    USER = "user"
    SYSTEM = "system"
    API = "api"
    SCHEDULER = "scheduler"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class AuditActor:
    """Represents the actor performing an audited action."""

    actor_type: ActorType
    actor_id: str
    actor_name: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.actor_type.value,
            "id": self.actor_id,
            "name": self.actor_name,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditActor":
        """Create from dictionary."""
        return cls(
            actor_type=ActorType(data.get("type", "system")),
            actor_id=data.get("id", "unknown"),
            actor_name=data.get("name"),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
        )


@dataclass
class AuditEntry:
    """A single audit log entry."""

    action: AuditAction
    actor: AuditActor
    resource_type: str
    resource_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    workspace_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    outcome: str = "success"  # success, failure, partial
    error_message: Optional[str] = None

    # Assigned by the system
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sequence_number: int = 0
    previous_hash: Optional[str] = None
    entry_hash: Optional[str] = None

    def compute_hash(self) -> str:
        """Compute cryptographic hash of the entry for tamper detection."""
        data = {
            "entry_id": self.entry_id,
            "action": self.action.value,
            "actor": self.actor.to_dict(),
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "timestamp": self.timestamp.isoformat(),
            "workspace_id": self.workspace_id,
            "details": self.details,
            "outcome": self.outcome,
            "error_message": self.error_message,
            "sequence_number": self.sequence_number,
            "previous_hash": self.previous_hash,
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "entry_id": self.entry_id,
            "action": self.action.value,
            "actor": self.actor.to_dict(),
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "timestamp": self.timestamp.isoformat(),
            "workspace_id": self.workspace_id,
            "details": self.details,
            "outcome": self.outcome,
            "error_message": self.error_message,
            "sequence_number": self.sequence_number,
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEntry":
        """Create from dictionary."""
        return cls(
            entry_id=data.get("entry_id", str(uuid.uuid4())),
            action=AuditAction(data["action"]),
            actor=AuditActor.from_dict(data.get("actor", {})),
            resource_type=data.get("resource_type", "unknown"),
            resource_id=data.get("resource_id", "unknown"),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if isinstance(data.get("timestamp"), str)
            else datetime.now(timezone.utc),
            workspace_id=data.get("workspace_id"),
            details=data.get("details", {}),
            outcome=data.get("outcome", "success"),
            error_message=data.get("error_message"),
            sequence_number=data.get("sequence_number", 0),
            previous_hash=data.get("previous_hash"),
            entry_hash=data.get("entry_hash"),
        )


@dataclass
class AuditQuery:
    """Query parameters for searching audit logs."""

    # Time range
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Filters
    actions: Optional[List[AuditAction]] = None
    actor_types: Optional[List[ActorType]] = None
    actor_ids: Optional[List[str]] = None
    resource_types: Optional[List[str]] = None
    resource_ids: Optional[List[str]] = None
    workspace_ids: Optional[List[str]] = None
    outcomes: Optional[List[str]] = None

    # Pagination
    limit: int = 100
    offset: int = 0

    def matches(self, entry: AuditEntry) -> bool:
        """Check if an entry matches this query."""
        # Time range
        if self.start_time and entry.timestamp < self.start_time:
            return False
        if self.end_time and entry.timestamp > self.end_time:
            return False

        # Action filter
        if self.actions and entry.action not in self.actions:
            return False

        # Actor filters
        if self.actor_types and entry.actor.actor_type not in self.actor_types:
            return False
        if self.actor_ids and entry.actor.actor_id not in self.actor_ids:
            return False

        # Resource filters
        if self.resource_types and entry.resource_type not in self.resource_types:
            return False
        if self.resource_ids and entry.resource_id not in self.resource_ids:
            return False

        # Workspace filter
        if self.workspace_ids and entry.workspace_id not in self.workspace_ids:
            return False

        # Outcome filter
        if self.outcomes and entry.outcome not in self.outcomes:
            return False

        return True


# =============================================================================
# Audit Log Storage
# =============================================================================


class AuditLog:
    """
    Immutable audit log with append-only storage.

    Uses Redis Streams for durability and ordering, with cryptographic
    hashing to detect tampering.

    Usage:
        audit = AuditLog(redis_url="redis://localhost:6379")
        await audit.connect()

        # Log an action
        await audit.log(
            action=AuditAction.TASK_COMPLETED,
            actor=AuditActor(ActorType.AGENT, "claude-3"),
            resource_type="task",
            resource_id="task-123",
            details={"result": "success"},
        )

        # Query logs
        entries = await audit.query(AuditQuery(
            actions=[AuditAction.TASK_COMPLETED],
            start_time=datetime.now() - timedelta(hours=1),
        ))
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        stream_key: str = AUDIT_STREAM_KEY,
        retention_days: int = AUDIT_RETENTION_DAYS,
    ):
        """Initialize the audit log."""
        self._redis_url = redis_url
        self._stream_key = stream_key
        self._retention_days = retention_days
        self._redis: Optional[Any] = None
        self._sequence_number = 0
        self._last_hash: Optional[str] = None

        # Local fallback storage
        self._local_entries: List[AuditEntry] = []

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self._redis.ping()

            # Get the last entry to continue the chain
            await self._initialize_chain()

            logger.info(
                "Audit log connected",
                extra={"redis_url": self._redis_url, "stream_key": self._stream_key},
            )

        except ImportError:
            try:
                from aragora.storage.production_guards import require_distributed_store, StorageMode
            except ImportError:
                pass
            else:
                require_distributed_store(
                    "control_plane_audit_log",
                    StorageMode.MEMORY,
                    "Redis not available for audit log",
                )
            logger.warning(
                "Redis not available, using in-memory audit log (not suitable for production)"
            )
            self._redis = None
        except Exception as e:
            try:
                from aragora.storage.production_guards import require_distributed_store, StorageMode
            except ImportError:
                pass
            else:
                require_distributed_store(
                    "control_plane_audit_log",
                    StorageMode.MEMORY,
                    f"Failed to connect to Redis for audit log: {e}",
                )
            logger.error(f"Failed to connect to Redis for audit log: {e}")
            self._redis = None

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            logger.info("Audit log disconnected")

    async def log(
        self,
        action: AuditAction,
        actor: AuditActor,
        resource_type: str,
        resource_id: str,
        workspace_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        outcome: str = "success",
        error_message: Optional[str] = None,
    ) -> AuditEntry:
        """
        Append an entry to the audit log.

        Args:
            action: The action being audited
            actor: Who performed the action
            resource_type: Type of resource affected
            resource_id: ID of resource affected
            workspace_id: Optional workspace context
            details: Additional action details
            outcome: Action outcome (success/failure/partial)
            error_message: Error details if failed

        Returns:
            The created AuditEntry
        """
        self._sequence_number += 1

        entry = AuditEntry(
            action=action,
            actor=actor,
            resource_type=resource_type,
            resource_id=resource_id,
            workspace_id=workspace_id,
            details=details or {},
            outcome=outcome,
            error_message=error_message,
            sequence_number=self._sequence_number,
            previous_hash=self._last_hash,
        )

        # Compute hash for tamper detection
        entry.entry_hash = entry.compute_hash()
        self._last_hash = entry.entry_hash

        # Store entry
        await self._store_entry(entry)

        logger.debug(
            f"Audit logged: {action.value}",
            extra={
                "entry_id": entry.entry_id,
                "actor": actor.actor_id,
                "resource": f"{resource_type}/{resource_id}",
            },
        )

        return entry

    async def query(self, query: AuditQuery) -> List[AuditEntry]:
        """
        Query the audit log.

        Args:
            query: Query parameters

        Returns:
            List of matching AuditEntry objects
        """
        if self._redis:
            return await self._query_redis(query)
        else:
            return self._query_local(query)

    async def verify_integrity(self, start_seq: int = 0, end_seq: Optional[int] = None) -> bool:
        """
        Verify the integrity of the audit log chain.

        Checks that hashes form an unbroken chain.

        Args:
            start_seq: Starting sequence number
            end_seq: Ending sequence number (None = latest)

        Returns:
            True if integrity verified, False if tampering detected
        """
        entries = await self.query(AuditQuery(limit=10000))

        if not entries:
            return True

        # Sort by sequence number
        entries.sort(key=lambda e: e.sequence_number)

        previous_hash = None
        for entry in entries:
            if entry.sequence_number < start_seq:
                previous_hash = entry.entry_hash
                continue

            if end_seq and entry.sequence_number > end_seq:
                break

            # Verify previous hash link
            if entry.previous_hash != previous_hash:
                logger.error(
                    f"Audit integrity violation at sequence {entry.sequence_number}: "
                    f"previous_hash mismatch"
                )
                return False

            # Verify entry hash
            computed_hash = entry.compute_hash()
            if entry.entry_hash != computed_hash:
                logger.error(
                    f"Audit integrity violation at sequence {entry.sequence_number}: "
                    f"entry_hash mismatch"
                )
                return False

            previous_hash = entry.entry_hash

        return True

    async def export(
        self,
        query: AuditQuery,
        format: str = "json",
    ) -> str:
        """
        Export audit logs in specified format.

        Args:
            query: Query to filter logs
            format: Export format (json, csv, syslog, soc2, iso27001)

        Returns:
            Exported data as string

        Supported formats:
        - json: Standard JSON array
        - csv: Comma-separated values
        - syslog: RFC 5424 syslog format
        - soc2: SOC 2 compliance report format
        - iso27001: ISO 27001 audit evidence format
        """
        entries = await self.query(query)

        if format == "json":
            return json.dumps([e.to_dict() for e in entries], indent=2, default=str)
        elif format == "csv":
            lines = [
                "entry_id,timestamp,action,actor_type,actor_id,resource_type,resource_id,outcome"
            ]
            for e in entries:
                lines.append(
                    f"{e.entry_id},{e.timestamp.isoformat()},{e.action.value},"
                    f"{e.actor.actor_type.value},{e.actor.actor_id},"
                    f"{e.resource_type},{e.resource_id},{e.outcome}"
                )
            return "\n".join(lines)
        elif format == "syslog":
            return self._export_syslog(entries)
        elif format == "soc2":
            return self._export_soc2(entries, query)
        elif format == "iso27001":
            return self._export_iso27001(entries, query)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_syslog(self, entries: List[AuditEntry]) -> str:
        """Export in RFC 5424 syslog format."""
        lines = []
        for e in entries:
            # RFC 5424: <priority>version timestamp hostname app-name procid msgid structured-data msg
            severity = 6 if e.outcome == "success" else 4  # INFO or WARNING
            facility = 10  # security/authorization
            priority = facility * 8 + severity

            structured_data = (
                f'[aragora@1 action="{e.action.value}" '
                f'actor="{e.actor.actor_id}" '
                f'resource="{e.resource_type}/{e.resource_id}" '
                f'outcome="{e.outcome}"]'
            )

            lines.append(
                f"<{priority}>1 {e.timestamp.isoformat()} aragora control-plane "
                f"{e.entry_id} - {structured_data} {e.action.value}: {e.outcome}"
            )

        return "\n".join(lines)

    def _export_soc2(self, entries: List[AuditEntry], query: AuditQuery) -> str:
        """Export in SOC 2 compliance report format."""
        # Group entries by control category
        controls: Dict[str, List[AuditEntry]] = {
            "CC6.1": [],  # Logical and physical access controls
            "CC6.2": [],  # System operations
            "CC6.3": [],  # Change management
            "CC7.1": [],  # Monitoring
            "CC7.2": [],  # Incident response
        }

        for e in entries:
            if e.action.value.startswith("auth."):
                controls["CC6.1"].append(e)
            elif e.action.value.startswith("agent.") or e.action.value.startswith("task."):
                controls["CC6.2"].append(e)
            elif e.action.value.startswith("config.") or e.action.value.startswith("policy."):
                controls["CC6.3"].append(e)
            elif e.action.value.startswith("deliberation."):
                controls["CC7.1"].append(e)
            elif e.action.value.startswith("system."):
                controls["CC7.2"].append(e)

        # Build report
        report = {
            "report_type": "SOC 2 Type II Audit Evidence",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period": {
                "start": query.start_time.isoformat() if query.start_time else None,
                "end": query.end_time.isoformat() if query.end_time else None,
            },
            "summary": {
                "total_events": len(entries),
                "by_outcome": {
                    "success": sum(1 for e in entries if e.outcome == "success"),
                    "failure": sum(1 for e in entries if e.outcome == "failure"),
                    "partial": sum(1 for e in entries if e.outcome == "partial"),
                },
            },
            "controls": {
                ctrl: {
                    "description": self._get_soc2_control_description(ctrl),
                    "event_count": len(events),
                    "events": [e.to_dict() for e in events[:100]],  # Limit per control
                }
                for ctrl, events in controls.items()
                if events
            },
        }

        return json.dumps(report, indent=2, default=str)

    def _get_soc2_control_description(self, control: str) -> str:
        """Get SOC 2 control description."""
        descriptions = {
            "CC6.1": "Logical and Physical Access Controls",
            "CC6.2": "System Operations",
            "CC6.3": "Change Management",
            "CC7.1": "System Monitoring",
            "CC7.2": "Incident Response",
        }
        return descriptions.get(control, "Unknown Control")

    def _export_iso27001(self, entries: List[AuditEntry], query: AuditQuery) -> str:
        """Export in ISO 27001 audit evidence format."""
        # Group by ISO 27001 control domains
        domains: Dict[str, List[AuditEntry]] = {
            "A.9": [],  # Access control
            "A.12": [],  # Operations security
            "A.14": [],  # System acquisition, development and maintenance
            "A.16": [],  # Information security incident management
            "A.18": [],  # Compliance
        }

        for e in entries:
            if e.action.value.startswith("auth."):
                domains["A.9"].append(e)
            elif e.action.value.startswith("task.") or e.action.value.startswith("agent."):
                domains["A.12"].append(e)
            elif e.action.value.startswith("config."):
                domains["A.14"].append(e)
            elif e.action.value.startswith("system.") or e.outcome == "failure":
                domains["A.16"].append(e)
            elif e.action.value.startswith("policy."):
                domains["A.18"].append(e)

        report = {
            "standard": "ISO/IEC 27001:2022",
            "report_type": "Audit Evidence Export",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "organization": "Aragora Control Plane",
            "audit_period": {
                "start": query.start_time.isoformat() if query.start_time else None,
                "end": query.end_time.isoformat() if query.end_time else None,
            },
            "executive_summary": {
                "total_log_entries": len(entries),
                "security_events": sum(1 for e in entries if e.outcome == "failure"),
                "compliance_rate": (
                    sum(1 for e in entries if e.outcome == "success") / len(entries) * 100
                    if entries
                    else 100
                ),
            },
            "control_domains": {
                domain: {
                    "name": self._get_iso27001_domain_name(domain),
                    "evidence_count": len(events),
                    "evidence_samples": [e.to_dict() for e in events[:50]],
                }
                for domain, events in domains.items()
                if events
            },
        }

        return json.dumps(report, indent=2, default=str)

    def _get_iso27001_domain_name(self, domain: str) -> str:
        """Get ISO 27001 domain name."""
        names = {
            "A.9": "Access Control",
            "A.12": "Operations Security",
            "A.14": "System Acquisition, Development and Maintenance",
            "A.16": "Information Security Incident Management",
            "A.18": "Compliance",
        }
        return names.get(domain, "Unknown Domain")

    async def enforce_retention(self) -> int:
        """
        Enforce retention policy by removing entries older than retention_days.

        Returns:
            Number of entries removed
        """
        if not self._redis:
            # Local mode: filter entries
            cutoff = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            from datetime import timedelta

            cutoff = cutoff - timedelta(days=self._retention_days)

            original_count = len(self._local_entries)
            self._local_entries = [e for e in self._local_entries if e.timestamp >= cutoff]
            removed = original_count - len(self._local_entries)
            logger.info(f"Retention enforcement removed {removed} local entries")
            return removed

        # Redis mode: use XTRIM with MINID
        try:
            from datetime import timedelta

            cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)
            cutoff_ms = int(cutoff.timestamp() * 1000)

            # Get count before
            info_before = await self._redis.xinfo_stream(self._stream_key)
            count_before = info_before.get("length", 0)

            # Trim entries older than cutoff
            await self._redis.xtrim(
                self._stream_key,
                minid=f"{cutoff_ms}-0",
            )

            # Get count after
            info_after = await self._redis.xinfo_stream(self._stream_key)
            count_after = info_after.get("length", 0)

            removed = count_before - count_after
            logger.info(f"Retention enforcement removed {removed} Redis entries")
            return removed

        except Exception as e:
            logger.error(f"Failed to enforce retention: {e}")
            return 0

    async def get_retention_status(self) -> Dict[str, Any]:
        """Get retention policy status."""
        total_entries = 0
        oldest_entry: Optional[datetime] = None

        if self._redis:
            try:
                info = await self._redis.xinfo_stream(self._stream_key)
                total_entries = info.get("length", 0)

                # Get oldest entry
                oldest = await self._redis.xrange(self._stream_key, count=1)
                if oldest:
                    _, data = oldest[0]
                    entry_data = json.loads(data.get("data", "{}"))
                    oldest_entry = datetime.fromisoformat(entry_data.get("timestamp", ""))
            except Exception as e:
                logger.error(f"Failed to get retention status: {e}")
        else:
            total_entries = len(self._local_entries)
            if self._local_entries:
                oldest_entry = min(e.timestamp for e in self._local_entries)

        from datetime import timedelta

        retention_cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)

        return {
            "retention_days": self._retention_days,
            "total_entries": total_entries,
            "oldest_entry": oldest_entry.isoformat() if oldest_entry else None,
            "retention_cutoff": retention_cutoff.isoformat(),
            "entries_eligible_for_removal": (
                sum(1 for e in self._local_entries if e.timestamp < retention_cutoff)
                if not self._redis
                else None
            ),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get audit log statistics."""
        return {
            "total_entries": self._sequence_number,
            "last_hash": self._last_hash,
            "storage_backend": "redis" if self._redis else "memory",
            "retention_days": self._retention_days,
        }

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _initialize_chain(self) -> None:
        """Initialize the hash chain from existing entries."""
        if not self._redis:
            return

        try:
            # Get the last entry from the stream
            entries = await self._redis.xrevrange(self._stream_key, count=1)
            if entries:
                _, data = entries[0]
                entry_data = json.loads(data.get("data", "{}"))
                self._sequence_number = entry_data.get("sequence_number", 0)
                self._last_hash = entry_data.get("entry_hash")
        except Exception as e:
            logger.error(f"Failed to initialize audit chain: {e}")

    async def _store_entry(self, entry: AuditEntry) -> None:
        """Store an entry in the log."""
        if self._redis:
            try:
                await self._redis.xadd(
                    self._stream_key,
                    {
                        "entry_id": entry.entry_id,
                        "data": json.dumps(entry.to_dict()),
                    },
                )
            except Exception as e:
                logger.error(f"Failed to store audit entry: {e}")
                # Fall back to local storage
                self._local_entries.append(entry)
        else:
            self._local_entries.append(entry)

    async def _query_redis(self, query: AuditQuery) -> List[AuditEntry]:
        """Query entries from Redis."""
        entries: List[AuditEntry] = []

        try:
            # Determine time range for Redis XRANGE
            start = "-"
            end = "+"

            if query.start_time:
                start = str(int(query.start_time.timestamp() * 1000))
            if query.end_time:
                end = str(int(query.end_time.timestamp() * 1000))

            # Fetch entries (Redis handles time-based filtering)
            raw_entries = await self._redis.xrange(
                self._stream_key,
                min=start,
                max=end,
                count=query.limit + query.offset + 1000,  # Fetch extra for filtering
            )

            for _, data in raw_entries:
                entry_data = json.loads(data.get("data", "{}"))
                entry = AuditEntry.from_dict(entry_data)

                if query.matches(entry):
                    entries.append(entry)

        except Exception as e:
            logger.error(f"Failed to query audit log: {e}")

        # Apply pagination
        return entries[query.offset : query.offset + query.limit]

    def _query_local(self, query: AuditQuery) -> List[AuditEntry]:
        """Query entries from local storage."""
        matching = [e for e in self._local_entries if query.matches(e)]
        return matching[query.offset : query.offset + query.limit]


# =============================================================================
# Helper Functions
# =============================================================================


def create_system_actor() -> AuditActor:
    """Create a system actor for automated actions."""
    return AuditActor(
        actor_type=ActorType.SYSTEM,
        actor_id="aragora-control-plane",
        actor_name="Aragora Control Plane",
    )


def create_agent_actor(agent_id: str, agent_name: Optional[str] = None) -> AuditActor:
    """Create an agent actor."""
    return AuditActor(
        actor_type=ActorType.AGENT,
        actor_id=agent_id,
        actor_name=agent_name or agent_id,
    )


def create_user_actor(
    user_id: str,
    user_name: Optional[str] = None,
    ip_address: Optional[str] = None,
) -> AuditActor:
    """Create a user actor."""
    return AuditActor(
        actor_type=ActorType.USER,
        actor_id=user_id,
        actor_name=user_name,
        ip_address=ip_address,
    )


# =============================================================================
# Convenience Logging Functions
# =============================================================================

# Global audit log instance for convenience functions
_audit_log: Optional[AuditLog] = None


def get_audit_log() -> Optional[AuditLog]:
    """Get the global audit log instance."""
    return _audit_log


def set_audit_log(audit_log: AuditLog) -> None:
    """Set the global audit log instance."""
    global _audit_log
    _audit_log = audit_log


async def log_policy_decision(
    policy_id: str,
    decision: str,  # "allow", "deny", "warn"
    task_type: str,
    reason: str,
    workspace_id: Optional[str] = None,
    task_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    violations: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[AuditEntry]:
    """
    Log a policy decision to the audit trail.

    Args:
        policy_id: ID of the policy that made the decision
        decision: Decision outcome ("allow", "deny", "warn")
        task_type: Type of task being evaluated
        reason: Human-readable reason for the decision
        workspace_id: Optional workspace context
        task_id: Optional task ID being evaluated
        agent_id: Optional agent involved
        violations: List of policy violations if any
        metadata: Additional metadata

    Returns:
        AuditEntry if logged, None if audit log not configured
    """
    if not _audit_log:
        logger.debug("Audit log not configured, skipping policy decision log")
        return None

    action_map = {
        "allow": AuditAction.POLICY_DECISION_ALLOW,
        "deny": AuditAction.POLICY_DECISION_DENY,
        "warn": AuditAction.POLICY_DECISION_WARN,
    }
    action = action_map.get(decision.lower(), AuditAction.POLICY_EVALUATED)

    details = {
        "task_type": task_type,
        "reason": reason,
        "violations": violations or [],
        **(metadata or {}),
    }

    if agent_id:
        details["agent_id"] = agent_id

    return await _audit_log.log(
        action=action,
        actor=create_system_actor(),
        resource_type="policy",
        resource_id=policy_id,
        workspace_id=workspace_id,
        details=details,
        outcome="success" if decision.lower() in ("allow", "warn") else "failure",
    )


async def log_deliberation_event(
    task_id: str,
    event_type: str,
    details: Dict[str, Any],
    workspace_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    outcome: str = "success",
    error_message: Optional[str] = None,
) -> Optional[AuditEntry]:
    """
    Log a deliberation event to the audit trail.

    Args:
        task_id: Deliberation task ID
        event_type: Type of deliberation event
        details: Event details
        workspace_id: Optional workspace context
        agent_id: Optional agent involved
        outcome: Event outcome ("success", "failure", "partial")
        error_message: Error message if failed

    Returns:
        AuditEntry if logged, None if audit log not configured
    """
    if not _audit_log:
        logger.debug("Audit log not configured, skipping deliberation event log")
        return None

    # Map event types to AuditAction
    event_action_map = {
        "started": AuditAction.DELIBERATION_STARTED,
        "round_completed": AuditAction.DELIBERATION_ROUND_COMPLETED,
        "consensus": AuditAction.DELIBERATION_CONSENSUS,
        "failed": AuditAction.DELIBERATION_FAILED,
        "timeout": AuditAction.DELIBERATION_TIMEOUT,
        "sla_warning": AuditAction.DELIBERATION_SLA_WARNING,
        "sla_critical": AuditAction.DELIBERATION_SLA_CRITICAL,
        "sla_violated": AuditAction.DELIBERATION_SLA_VIOLATED,
    }

    action = event_action_map.get(event_type, AuditAction.DELIBERATION_STARTED)

    if agent_id:
        actor = create_agent_actor(agent_id)
    else:
        actor = create_system_actor()

    return await _audit_log.log(
        action=action,
        actor=actor,
        resource_type="deliberation",
        resource_id=task_id,
        workspace_id=workspace_id,
        details=details,
        outcome=outcome,
        error_message=error_message,
    )


async def log_deliberation_started(
    task_id: str,
    question: str,
    agents: List[str],
    sla_timeout_seconds: float,
    workspace_id: Optional[str] = None,
) -> Optional[AuditEntry]:
    """Log deliberation start event."""
    return await log_deliberation_event(
        task_id=task_id,
        event_type="started",
        details={
            "question_preview": question[:200] if question else "",
            "agents": agents,
            "sla_timeout_seconds": sla_timeout_seconds,
        },
        workspace_id=workspace_id,
    )


async def log_deliberation_completed(
    task_id: str,
    success: bool,
    consensus_reached: bool,
    confidence: float,
    duration_seconds: float,
    sla_compliant: bool,
    workspace_id: Optional[str] = None,
    winner: Optional[str] = None,
) -> Optional[AuditEntry]:
    """Log deliberation completion event."""
    return await log_deliberation_event(
        task_id=task_id,
        event_type="consensus" if consensus_reached else "failed",
        details={
            "success": success,
            "consensus_reached": consensus_reached,
            "confidence": confidence,
            "duration_seconds": duration_seconds,
            "sla_compliant": sla_compliant,
            "winner": winner,
        },
        workspace_id=workspace_id,
        outcome="success" if success else "failure",
    )


async def log_deliberation_sla_event(
    task_id: str,
    level: str,  # "warning", "critical", "violated"
    elapsed_seconds: float,
    timeout_seconds: float,
    workspace_id: Optional[str] = None,
) -> Optional[AuditEntry]:
    """Log deliberation SLA event."""
    return await log_deliberation_event(
        task_id=task_id,
        event_type=f"sla_{level}",
        details={
            "elapsed_seconds": elapsed_seconds,
            "timeout_seconds": timeout_seconds,
            "remaining_seconds": timeout_seconds - elapsed_seconds,
            "pct_used": (elapsed_seconds / timeout_seconds * 100) if timeout_seconds > 0 else 100,
        },
        workspace_id=workspace_id,
        outcome="partial" if level in ("warning", "critical") else "failure",
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "AuditAction",
    "ActorType",
    # Data Classes
    "AuditActor",
    "AuditEntry",
    "AuditQuery",
    # Main Class
    "AuditLog",
    # Helpers
    "create_system_actor",
    "create_agent_actor",
    "create_user_actor",
    # Global audit log
    "get_audit_log",
    "set_audit_log",
    # Convenience functions
    "log_policy_decision",
    "log_deliberation_event",
    "log_deliberation_started",
    "log_deliberation_completed",
    "log_deliberation_sla_event",
]
