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
            logger.warning(
                "Redis not available, using in-memory audit log (not suitable for production)"
            )
            self._redis = None
        except Exception as e:
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
            format: Export format (json, csv)

        Returns:
            Exported data as string
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
        else:
            raise ValueError(f"Unsupported export format: {format}")

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
]
