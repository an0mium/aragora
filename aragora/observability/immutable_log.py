"""
Immutable Audit Logging System.

Provides append-only, tamper-evident audit trails for compliance requirements.
Supports multiple backends: local file, S3 Object Lock, AWS QLDB.

Features:
- Hash chain verification (blockchain-style)
- Cryptographic tamper detection
- Multiple storage backends
- Query and export capabilities
- Daily hash anchors for external verification

Usage:
    from aragora.observability.immutable_log import (
        ImmutableAuditLog,
        get_audit_log,
        AuditEntry,
        AuditBackend,
    )

    # Get the audit log
    log = get_audit_log()

    # Append an entry
    entry = await log.append(
        event_type="finding_created",
        actor="user@example.com",
        resource_type="finding",
        resource_id="f-123",
        action="create",
        details={"severity": "high"},
    )

    # Verify integrity
    is_valid, errors = await log.verify_integrity()

    # Query entries
    entries = await log.query(
        start_time=datetime.now() - timedelta(days=7),
        event_types=["finding_created", "finding_updated"],
    )

    # Export for compliance
    await log.export_range(start, end, format="json", path="/tmp/audit.json")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import threading
import uuid

logger = logging.getLogger(__name__)


class AuditBackend(str, Enum):
    """Supported audit log backends."""

    LOCAL = "local"  # Local append-only file (good for dev/testing)
    S3_OBJECT_LOCK = "s3_object_lock"  # S3 with Object Lock (WORM compliance)
    QLDB = "qldb"  # AWS QLDB (cryptographic verification, queryable)


@dataclass
class AuditEntry:
    """A single immutable audit log entry."""

    # Unique identifier
    id: str

    # Timestamp (UTC)
    timestamp: datetime

    # Hash chain fields
    sequence_number: int
    previous_hash: str
    entry_hash: str

    # Event data
    event_type: str
    actor: str  # User ID or system identifier
    actor_type: str  # "user", "system", "agent"
    resource_type: str  # "finding", "document", "audit_session", etc.
    resource_id: str
    action: str  # "create", "update", "delete", "access", etc.

    # Additional context
    details: dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    workspace_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # Integrity
    signature: Optional[str] = None  # Optional cryptographic signature

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "sequence_number": self.sequence_number,
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
            "event_type": self.event_type,
            "actor": self.actor,
            "actor_type": self.actor_type,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "details": self.details,
            "correlation_id": self.correlation_id,
            "workspace_id": self.workspace_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditEntry:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            sequence_number=data["sequence_number"],
            previous_hash=data["previous_hash"],
            entry_hash=data["entry_hash"],
            event_type=data["event_type"],
            actor=data["actor"],
            actor_type=data.get("actor_type", "user"),
            resource_type=data["resource_type"],
            resource_id=data["resource_id"],
            action=data["action"],
            details=data.get("details", {}),
            correlation_id=data.get("correlation_id"),
            workspace_id=data.get("workspace_id"),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            signature=data.get("signature"),
        )

    def compute_hash(self) -> str:
        """Compute the hash of this entry's content."""
        content = {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "sequence_number": self.sequence_number,
            "previous_hash": self.previous_hash,
            "event_type": self.event_type,
            "actor": self.actor,
            "actor_type": self.actor_type,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "details": self.details,
            "correlation_id": self.correlation_id,
            "workspace_id": self.workspace_id,
        }
        content_bytes = json.dumps(content, sort_keys=True).encode("utf-8")
        return hashlib.sha256(content_bytes).hexdigest()


@dataclass
class DailyAnchor:
    """Daily hash anchor for external verification."""

    date: str  # YYYY-MM-DD
    first_sequence: int
    last_sequence: int
    entry_count: int
    merkle_root: str  # Root of Merkle tree for day's entries
    chain_hash: str  # Hash of last entry in day
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date,
            "first_sequence": self.first_sequence,
            "last_sequence": self.last_sequence,
            "entry_count": self.entry_count,
            "merkle_root": self.merkle_root,
            "chain_hash": self.chain_hash,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class VerificationResult:
    """Result of integrity verification."""

    is_valid: bool
    entries_checked: int
    errors: list[str]
    warnings: list[str]
    first_error_sequence: Optional[int] = None
    verification_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "entries_checked": self.entries_checked,
            "errors": self.errors,
            "warnings": self.warnings,
            "first_error_sequence": self.first_error_sequence,
            "verification_time_ms": self.verification_time_ms,
        }


class AuditLogBackend(ABC):
    """Abstract base class for audit log storage backends."""

    @abstractmethod
    async def append(self, entry: AuditEntry) -> None:
        """Append an entry to the log."""
        ...

    @abstractmethod
    async def get_entry(self, entry_id: str) -> Optional[AuditEntry]:
        """Get a specific entry by ID."""
        ...

    @abstractmethod
    async def get_by_sequence(self, sequence_number: int) -> Optional[AuditEntry]:
        """Get entry by sequence number."""
        ...

    @abstractmethod
    async def get_last_entry(self) -> Optional[AuditEntry]:
        """Get the most recent entry."""
        ...

    @abstractmethod
    async def get_entries_range(
        self,
        start_sequence: int,
        end_sequence: int,
    ) -> list[AuditEntry]:
        """Get entries in a sequence range."""
        ...

    @abstractmethod
    async def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[list[str]] = None,
        actors: Optional[list[str]] = None,
        resource_types: Optional[list[str]] = None,
        resource_ids: Optional[list[str]] = None,
        workspace_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEntry]:
        """Query entries with filters."""
        ...

    @abstractmethod
    async def count(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[list[str]] = None,
    ) -> int:
        """Count entries matching filters."""
        ...

    @abstractmethod
    async def save_anchor(self, anchor: DailyAnchor) -> None:
        """Save a daily anchor."""
        ...

    @abstractmethod
    async def get_anchor(self, date: str) -> Optional[DailyAnchor]:
        """Get anchor for a specific date."""
        ...


class LocalFileBackend(AuditLogBackend):
    """
    Local file-based backend for development and testing.

    Uses append-only file writes with JSON lines format.
    Maintains an index file for fast lookups.
    """

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / "audit.jsonl"
        self.index_file = self.log_dir / "index.json"
        self.anchors_file = self.log_dir / "anchors.json"

        self._lock = threading.Lock()
        self._index: dict[str, int] = {}  # entry_id -> file_offset
        self._sequence_index: dict[int, int] = {}  # sequence -> file_offset
        self._load_index()

    def _load_index(self) -> None:
        """Load index from file."""
        if self.index_file.exists():
            try:
                data = json.loads(self.index_file.read_text())
                self._index = data.get("entries", {})
                self._sequence_index = {int(k): v for k, v in data.get("sequences", {}).items()}
            except Exception as e:
                logger.warning(f"Failed to load audit index: {e}")
                self._rebuild_index()
        elif self.log_file.exists():
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild index from log file."""
        self._index = {}
        self._sequence_index = {}

        if not self.log_file.exists():
            return

        try:
            with open(self.log_file, "r") as f:
                offset = 0
                for line in f:
                    if line.strip():
                        entry_data = json.loads(line)
                        self._index[entry_data["id"]] = offset
                        self._sequence_index[entry_data["sequence_number"]] = offset
                    offset = f.tell()

            self._save_index()
        except Exception as e:
            logger.error(f"Failed to rebuild audit index: {e}")

    def _save_index(self) -> None:
        """Save index to file."""
        try:
            data = {
                "entries": self._index,
                "sequences": {str(k): v for k, v in self._sequence_index.items()},
            }
            self.index_file.write_text(json.dumps(data))
        except Exception as e:
            logger.error(f"Failed to save audit index: {e}")

    async def append(self, entry: AuditEntry) -> None:
        """Append entry to log file."""
        with self._lock:
            # Append to log file
            with open(self.log_file, "a") as f:
                offset = f.tell()
                f.write(json.dumps(entry.to_dict()) + "\n")

            # Update indices
            self._index[entry.id] = offset
            self._sequence_index[entry.sequence_number] = offset
            self._save_index()

    async def get_entry(self, entry_id: str) -> Optional[AuditEntry]:
        """Get entry by ID."""
        offset = self._index.get(entry_id)
        if offset is None:
            return None

        try:
            with open(self.log_file, "r") as f:
                f.seek(offset)
                line = f.readline()
                if line:
                    return AuditEntry.from_dict(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read audit entry {entry_id}: {e}")

        return None

    async def get_by_sequence(self, sequence_number: int) -> Optional[AuditEntry]:
        """Get entry by sequence number."""
        offset = self._sequence_index.get(sequence_number)
        if offset is None:
            return None

        try:
            with open(self.log_file, "r") as f:
                f.seek(offset)
                line = f.readline()
                if line:
                    return AuditEntry.from_dict(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read audit entry seq={sequence_number}: {e}")

        return None

    async def get_last_entry(self) -> Optional[AuditEntry]:
        """Get most recent entry."""
        if not self._sequence_index:
            return None

        max_seq = max(self._sequence_index.keys())
        return await self.get_by_sequence(max_seq)

    async def get_entries_range(
        self,
        start_sequence: int,
        end_sequence: int,
    ) -> list[AuditEntry]:
        """Get entries in sequence range."""
        entries = []
        for seq in range(start_sequence, end_sequence + 1):
            entry = await self.get_by_sequence(seq)
            if entry:
                entries.append(entry)
        return entries

    async def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[list[str]] = None,
        actors: Optional[list[str]] = None,
        resource_types: Optional[list[str]] = None,
        resource_ids: Optional[list[str]] = None,
        workspace_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEntry]:
        """Query entries with filters."""
        if not self.log_file.exists():
            return []

        results = []
        skipped = 0

        with open(self.log_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    entry = AuditEntry.from_dict(data)

                    # Apply filters
                    if start_time and entry.timestamp < start_time:
                        continue
                    if end_time and entry.timestamp > end_time:
                        continue
                    if event_types and entry.event_type not in event_types:
                        continue
                    if actors and entry.actor not in actors:
                        continue
                    if resource_types and entry.resource_type not in resource_types:
                        continue
                    if resource_ids and entry.resource_id not in resource_ids:
                        continue
                    if workspace_id and entry.workspace_id != workspace_id:
                        continue

                    # Apply offset
                    if skipped < offset:
                        skipped += 1
                        continue

                    results.append(entry)

                    if len(results) >= limit:
                        break

                except Exception as e:
                    logger.warning(f"Failed to parse audit entry: {e}")
                    continue

        return results

    async def count(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[list[str]] = None,
    ) -> int:
        """Count entries matching filters."""
        if not self.log_file.exists():
            return 0

        count = 0
        with open(self.log_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    entry = AuditEntry.from_dict(data)

                    if start_time and entry.timestamp < start_time:
                        continue
                    if end_time and entry.timestamp > end_time:
                        continue
                    if event_types and entry.event_type not in event_types:
                        continue

                    count += 1

                except Exception:
                    continue

        return count

    async def save_anchor(self, anchor: DailyAnchor) -> None:
        """Save daily anchor."""
        anchors = {}
        if self.anchors_file.exists():
            try:
                anchors = json.loads(self.anchors_file.read_text())
            except Exception:
                pass

        anchors[anchor.date] = anchor.to_dict()
        self.anchors_file.write_text(json.dumps(anchors, indent=2))

    async def get_anchor(self, date: str) -> Optional[DailyAnchor]:
        """Get anchor for date."""
        if not self.anchors_file.exists():
            return None

        try:
            anchors = json.loads(self.anchors_file.read_text())
            if date in anchors:
                data = anchors[date]
                return DailyAnchor(
                    date=data["date"],
                    first_sequence=data["first_sequence"],
                    last_sequence=data["last_sequence"],
                    entry_count=data["entry_count"],
                    merkle_root=data["merkle_root"],
                    chain_hash=data["chain_hash"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                )
        except Exception as e:
            logger.error(f"Failed to get anchor for {date}: {e}")

        return None


class S3ObjectLockBackend(AuditLogBackend):
    """
    S3 backend with Object Lock for WORM compliance.

    Stores entries as individual objects with retention policies.
    Requires S3 bucket with Object Lock enabled.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "audit-logs/",
        region: str = "us-east-1",
        retention_days: int = 2555,  # 7 years for compliance
    ):
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self.retention_days = retention_days
        self._client: Any = None
        self._sequence_cache: dict[int, str] = {}  # sequence -> object_key

    def _get_client(self) -> Any:
        """Get or create S3 client."""
        if self._client is None:
            try:
                import boto3

                self._client = boto3.client("s3", region_name=self.region)
            except ImportError:
                raise ImportError("boto3 required for S3 backend: pip install boto3")
        return self._client

    def _entry_key(self, entry: AuditEntry) -> str:
        """Generate S3 key for entry."""
        date_prefix = entry.timestamp.strftime("%Y/%m/%d")
        return f"{self.prefix}{date_prefix}/{entry.sequence_number:012d}_{entry.id}.json"

    async def append(self, entry: AuditEntry) -> None:
        """Append entry to S3 with Object Lock."""
        client = self._get_client()
        key = self._entry_key(entry)

        # Calculate retention date
        retention_date = datetime.now(timezone.utc) + timedelta(days=self.retention_days)

        try:
            client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=json.dumps(entry.to_dict()),
                ContentType="application/json",
                ObjectLockMode="GOVERNANCE",
                ObjectLockRetainUntilDate=retention_date,
            )
            self._sequence_cache[entry.sequence_number] = key
        except Exception as e:
            logger.error(f"Failed to write audit entry to S3: {e}")
            raise

    async def get_entry(self, entry_id: str) -> Optional[AuditEntry]:
        """Get entry by ID (requires listing)."""
        client = self._get_client()

        try:
            # List objects to find the entry
            paginator = client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
                for obj in page.get("Contents", []):
                    if entry_id in obj["Key"]:
                        response = client.get_object(Bucket=self.bucket, Key=obj["Key"])
                        data = json.loads(response["Body"].read())
                        return AuditEntry.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to get audit entry {entry_id} from S3: {e}")

        return None

    async def get_by_sequence(self, sequence_number: int) -> Optional[AuditEntry]:
        """Get entry by sequence number."""
        client = self._get_client()

        # Check cache
        if sequence_number in self._sequence_cache:
            try:
                response = client.get_object(
                    Bucket=self.bucket,
                    Key=self._sequence_cache[sequence_number],
                )
                data = json.loads(response["Body"].read())
                return AuditEntry.from_dict(data)
            except Exception:
                del self._sequence_cache[sequence_number]

        # Search by prefix pattern
        seq_prefix = f"{sequence_number:012d}_"
        try:
            paginator = client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
                for obj in page.get("Contents", []):
                    if seq_prefix in obj["Key"]:
                        response = client.get_object(Bucket=self.bucket, Key=obj["Key"])
                        data = json.loads(response["Body"].read())
                        self._sequence_cache[sequence_number] = obj["Key"]
                        return AuditEntry.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to get audit entry seq={sequence_number} from S3: {e}")

        return None

    async def get_last_entry(self) -> Optional[AuditEntry]:
        """Get most recent entry."""
        client = self._get_client()

        try:
            # List objects in reverse order by key
            paginator = client.get_paginator("list_objects_v2")
            last_key = None
            for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
                for obj in page.get("Contents", []):
                    if obj["Key"].endswith(".json"):
                        last_key = obj["Key"]

            if last_key:
                response = client.get_object(Bucket=self.bucket, Key=last_key)
                data = json.loads(response["Body"].read())
                return AuditEntry.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to get last audit entry from S3: {e}")

        return None

    async def get_entries_range(
        self,
        start_sequence: int,
        end_sequence: int,
    ) -> list[AuditEntry]:
        """Get entries in sequence range."""
        entries = []
        for seq in range(start_sequence, end_sequence + 1):
            entry = await self.get_by_sequence(seq)
            if entry:
                entries.append(entry)
        return entries

    async def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[list[str]] = None,
        actors: Optional[list[str]] = None,
        resource_types: Optional[list[str]] = None,
        resource_ids: Optional[list[str]] = None,
        workspace_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEntry]:
        """Query entries with filters."""
        client = self._get_client()
        results = []
        skipped = 0

        # Build date prefixes for efficient listing
        prefixes = [self.prefix]
        if start_time and end_time:
            # Generate date prefixes in range
            prefixes = []
            current = start_time.date()
            end = end_time.date()
            while current <= end:
                prefixes.append(f"{self.prefix}{current.strftime('%Y/%m/%d')}/")
                current += timedelta(days=1)

        try:
            paginator = client.get_paginator("list_objects_v2")
            for prefix in prefixes:
                for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                    for obj in page.get("Contents", []):
                        if not obj["Key"].endswith(".json"):
                            continue

                        response = client.get_object(Bucket=self.bucket, Key=obj["Key"])
                        data = json.loads(response["Body"].read())
                        entry = AuditEntry.from_dict(data)

                        # Apply filters
                        if start_time and entry.timestamp < start_time:
                            continue
                        if end_time and entry.timestamp > end_time:
                            continue
                        if event_types and entry.event_type not in event_types:
                            continue
                        if actors and entry.actor not in actors:
                            continue
                        if resource_types and entry.resource_type not in resource_types:
                            continue
                        if resource_ids and entry.resource_id not in resource_ids:
                            continue
                        if workspace_id and entry.workspace_id != workspace_id:
                            continue

                        if skipped < offset:
                            skipped += 1
                            continue

                        results.append(entry)
                        if len(results) >= limit:
                            return results

        except Exception as e:
            logger.error(f"Failed to query audit entries from S3: {e}")

        return results

    async def count(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[list[str]] = None,
    ) -> int:
        """Count entries matching filters."""
        client = self._get_client()
        count = 0

        try:
            paginator = client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
                for obj in page.get("Contents", []):
                    if not obj["Key"].endswith(".json"):
                        continue

                    response = client.get_object(Bucket=self.bucket, Key=obj["Key"])
                    data = json.loads(response["Body"].read())
                    entry = AuditEntry.from_dict(data)

                    if start_time and entry.timestamp < start_time:
                        continue
                    if end_time and entry.timestamp > end_time:
                        continue
                    if event_types and entry.event_type not in event_types:
                        continue

                    count += 1

        except Exception as e:
            logger.error(f"Failed to count audit entries from S3: {e}")

        return count

    async def save_anchor(self, anchor: DailyAnchor) -> None:
        """Save daily anchor to S3."""
        client = self._get_client()
        key = f"{self.prefix}anchors/{anchor.date}.json"

        try:
            client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=json.dumps(anchor.to_dict()),
                ContentType="application/json",
            )
        except Exception as e:
            logger.error(f"Failed to save anchor to S3: {e}")
            raise

    async def get_anchor(self, date: str) -> Optional[DailyAnchor]:
        """Get anchor from S3."""
        client = self._get_client()
        key = f"{self.prefix}anchors/{date}.json"

        try:
            response = client.get_object(Bucket=self.bucket, Key=key)
            data = json.loads(response["Body"].read())
            return DailyAnchor(
                date=data["date"],
                first_sequence=data["first_sequence"],
                last_sequence=data["last_sequence"],
                entry_count=data["entry_count"],
                merkle_root=data["merkle_root"],
                chain_hash=data["chain_hash"],
                created_at=datetime.fromisoformat(data["created_at"]),
            )
        except client.exceptions.NoSuchKey:
            return None
        except Exception as e:
            logger.error(f"Failed to get anchor from S3: {e}")
            return None


class ImmutableAuditLog:
    """
    Main interface for immutable audit logging.

    Provides append-only, tamper-evident audit trails with:
    - Hash chain verification
    - Multiple storage backends
    - Daily anchors for external verification
    - Query and export capabilities
    """

    # Genesis hash for the first entry
    GENESIS_HASH = "0" * 64

    def __init__(
        self,
        backend: AuditLogBackend,
        signing_key: Optional[bytes] = None,
    ):
        self.backend = backend
        self.signing_key = signing_key
        self._lock = asyncio.Lock()
        self._last_entry: Optional[AuditEntry] = None
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Load last entry for hash chain continuity."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            self._last_entry = await self.backend.get_last_entry()
            self._initialized = True

    def _compute_merkle_root(self, hashes: list[str]) -> str:
        """Compute Merkle root from list of hashes."""
        if not hashes:
            return self.GENESIS_HASH

        # Pad to power of 2
        while len(hashes) & (len(hashes) - 1) != 0:
            hashes.append(hashes[-1])

        # Build tree
        while len(hashes) > 1:
            next_level = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + hashes[i + 1]
                next_level.append(hashlib.sha256(combined.encode()).hexdigest())
            hashes = next_level

        return hashes[0]

    def _sign_entry(self, entry: AuditEntry) -> Optional[str]:
        """Sign entry hash with HMAC if signing key is configured."""
        if not self.signing_key:
            return None

        import hmac

        signature = hmac.new(
            self.signing_key,
            entry.entry_hash.encode(),
            hashlib.sha256,
        ).hexdigest()

        return signature

    async def append(
        self,
        event_type: str,
        actor: str,
        resource_type: str,
        resource_id: str,
        action: str,
        actor_type: str = "user",
        details: Optional[dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuditEntry:
        """
        Append a new entry to the immutable log.

        Args:
            event_type: Type of event (e.g., "finding_created")
            actor: User ID or system identifier
            resource_type: Type of resource (e.g., "finding", "document")
            resource_id: ID of the affected resource
            action: Action performed (e.g., "create", "update")
            actor_type: Type of actor ("user", "system", "agent")
            details: Additional event details
            correlation_id: Request correlation ID
            workspace_id: Workspace ID
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            The created audit entry
        """
        await self._ensure_initialized()

        async with self._lock:
            # Determine sequence and previous hash
            if self._last_entry:
                sequence = self._last_entry.sequence_number + 1
                previous_hash = self._last_entry.entry_hash
            else:
                sequence = 1
                previous_hash = self.GENESIS_HASH

            # Create entry
            entry = AuditEntry(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                sequence_number=sequence,
                previous_hash=previous_hash,
                entry_hash="",  # Computed below
                event_type=event_type,
                actor=actor,
                actor_type=actor_type,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
                details=details or {},
                correlation_id=correlation_id,
                workspace_id=workspace_id,
                ip_address=ip_address,
                user_agent=user_agent,
            )

            # Compute and set hash
            entry.entry_hash = entry.compute_hash()

            # Sign if configured
            entry.signature = self._sign_entry(entry)

            # Persist
            await self.backend.append(entry)
            self._last_entry = entry

            logger.debug(
                f"Audit entry appended: seq={sequence} type={event_type} "
                f"resource={resource_type}/{resource_id}"
            )

            return entry

    async def verify_integrity(
        self,
        start_sequence: Optional[int] = None,
        end_sequence: Optional[int] = None,
    ) -> VerificationResult:
        """
        Verify the integrity of the hash chain.

        Args:
            start_sequence: Start of range to verify (default: 1)
            end_sequence: End of range (default: last entry)

        Returns:
            Verification result with any errors found
        """
        await self._ensure_initialized()

        start_time = time.time()
        errors: list[str] = []
        warnings: list[str] = []
        first_error_seq: Optional[int] = None

        # Determine range
        if start_sequence is None:
            start_sequence = 1
        if end_sequence is None:
            if self._last_entry:
                end_sequence = self._last_entry.sequence_number
            else:
                return VerificationResult(
                    is_valid=True,
                    entries_checked=0,
                    errors=[],
                    warnings=["No entries to verify"],
                    verification_time_ms=0,
                )

        # Verify entries
        entries = await self.backend.get_entries_range(start_sequence, end_sequence)

        if not entries:
            return VerificationResult(
                is_valid=True,
                entries_checked=0,
                errors=[],
                warnings=["No entries in range"],
                verification_time_ms=(time.time() - start_time) * 1000,
            )

        # Get previous entry for chain verification
        expected_prev_hash = self.GENESIS_HASH
        if start_sequence > 1:
            prev_entry = await self.backend.get_by_sequence(start_sequence - 1)
            if prev_entry:
                expected_prev_hash = prev_entry.entry_hash
            else:
                warnings.append(f"Could not find entry {start_sequence - 1} for chain verification")

        # Verify each entry
        for entry in entries:
            # Verify hash chain
            if entry.previous_hash != expected_prev_hash:
                error = (
                    f"Chain broken at seq={entry.sequence_number}: "
                    f"expected prev_hash={expected_prev_hash[:16]}..., "
                    f"got={entry.previous_hash[:16]}..."
                )
                errors.append(error)
                if first_error_seq is None:
                    first_error_seq = entry.sequence_number

            # Verify entry hash
            computed_hash = entry.compute_hash()
            if entry.entry_hash != computed_hash:
                error = (
                    f"Hash mismatch at seq={entry.sequence_number}: "
                    f"stored={entry.entry_hash[:16]}..., "
                    f"computed={computed_hash[:16]}..."
                )
                errors.append(error)
                if first_error_seq is None:
                    first_error_seq = entry.sequence_number

            # Verify signature if present
            if entry.signature and self.signing_key:
                expected_sig = self._sign_entry(entry)
                if entry.signature != expected_sig:
                    error = f"Invalid signature at seq={entry.sequence_number}"
                    errors.append(error)
                    if first_error_seq is None:
                        first_error_seq = entry.sequence_number

            expected_prev_hash = entry.entry_hash

        elapsed_ms = (time.time() - start_time) * 1000

        return VerificationResult(
            is_valid=len(errors) == 0,
            entries_checked=len(entries),
            errors=errors,
            warnings=warnings,
            first_error_sequence=first_error_seq,
            verification_time_ms=elapsed_ms,
        )

    async def create_daily_anchor(self, date: Optional[str] = None) -> Optional[DailyAnchor]:
        """
        Create a daily anchor for external verification.

        Args:
            date: Date in YYYY-MM-DD format (default: yesterday)

        Returns:
            The created anchor, or None if no entries for date
        """
        if date is None:
            date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

        # Parse date
        anchor_date = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_time = anchor_date
        end_time = anchor_date + timedelta(days=1)

        # Get entries for the day
        entries = await self.backend.query(
            start_time=start_time,
            end_time=end_time,
            limit=100000,  # Large limit for full day
        )

        if not entries:
            return None

        # Compute Merkle root
        entry_hashes = [e.entry_hash for e in entries]
        merkle_root = self._compute_merkle_root(entry_hashes)

        # Create anchor
        anchor = DailyAnchor(
            date=date,
            first_sequence=entries[0].sequence_number,
            last_sequence=entries[-1].sequence_number,
            entry_count=len(entries),
            merkle_root=merkle_root,
            chain_hash=entries[-1].entry_hash,
            created_at=datetime.now(timezone.utc),
        )

        await self.backend.save_anchor(anchor)
        logger.info(f"Created daily anchor for {date}: {len(entries)} entries")

        return anchor

    async def verify_anchor(self, date: str) -> VerificationResult:
        """
        Verify entries against a daily anchor.

        Args:
            date: Date to verify

        Returns:
            Verification result
        """
        anchor = await self.backend.get_anchor(date)
        if not anchor:
            return VerificationResult(
                is_valid=False,
                entries_checked=0,
                errors=[f"No anchor found for {date}"],
                warnings=[],
            )

        # Get entries for the anchor range
        entries = await self.backend.get_entries_range(
            anchor.first_sequence,
            anchor.last_sequence,
        )

        errors = []
        warnings = []

        # Verify entry count
        if len(entries) != anchor.entry_count:
            errors.append(
                f"Entry count mismatch: expected {anchor.entry_count}, got {len(entries)}"
            )

        # Verify Merkle root
        entry_hashes = [e.entry_hash for e in entries]
        computed_root = self._compute_merkle_root(entry_hashes)
        if computed_root != anchor.merkle_root:
            errors.append(
                f"Merkle root mismatch: expected {anchor.merkle_root[:16]}..., "
                f"got {computed_root[:16]}..."
            )

        # Verify chain hash
        if entries and entries[-1].entry_hash != anchor.chain_hash:
            errors.append(
                f"Chain hash mismatch: expected {anchor.chain_hash[:16]}..., "
                f"got {entries[-1].entry_hash[:16]}..."
            )

        # Also verify the chain integrity
        chain_result = await self.verify_integrity(
            anchor.first_sequence,
            anchor.last_sequence,
        )

        errors.extend(chain_result.errors)
        warnings.extend(chain_result.warnings)

        return VerificationResult(
            is_valid=len(errors) == 0,
            entries_checked=len(entries),
            errors=errors,
            warnings=warnings,
            verification_time_ms=chain_result.verification_time_ms,
        )

    async def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[list[str]] = None,
        actors: Optional[list[str]] = None,
        resource_types: Optional[list[str]] = None,
        resource_ids: Optional[list[str]] = None,
        workspace_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEntry]:
        """
        Query audit entries with filters.

        Args:
            start_time: Start of time range
            end_time: End of time range
            event_types: Filter by event types
            actors: Filter by actors
            resource_types: Filter by resource types
            resource_ids: Filter by resource IDs
            workspace_id: Filter by workspace
            limit: Maximum entries to return
            offset: Number of entries to skip

        Returns:
            List of matching entries
        """
        return await self.backend.query(
            start_time=start_time,
            end_time=end_time,
            event_types=event_types,
            actors=actors,
            resource_types=resource_types,
            resource_ids=resource_ids,
            workspace_id=workspace_id,
            limit=limit,
            offset=offset,
        )

    async def export_range(
        self,
        start_time: datetime,
        end_time: datetime,
        format: str = "json",
        path: Optional[str] = None,
    ) -> str | bytes:
        """
        Export audit entries for compliance reporting.

        Args:
            start_time: Start of export range
            end_time: End of export range
            format: Output format ("json", "csv", "jsonl")
            path: Optional file path to write to

        Returns:
            Exported data as string/bytes, or path if written to file
        """
        entries = await self.backend.query(
            start_time=start_time,
            end_time=end_time,
            limit=1000000,  # Large limit for export
        )

        if format == "json":
            data = json.dumps(
                {
                    "export_time": datetime.now(timezone.utc).isoformat(),
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "entry_count": len(entries),
                    "entries": [e.to_dict() for e in entries],
                },
                indent=2,
            )
        elif format == "jsonl":
            lines = [json.dumps(e.to_dict()) for e in entries]
            data = "\n".join(lines)
        elif format == "csv":
            import csv
            import io

            output = io.StringIO()
            if entries:
                writer = csv.DictWriter(output, fieldnames=entries[0].to_dict().keys())
                writer.writeheader()
                for entry in entries:
                    writer.writerow(entry.to_dict())
            data = output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")

        if path:
            Path(path).write_text(data)
            return path

        return data

    async def get_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """
        Get audit log statistics.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Statistics dictionary
        """
        await self._ensure_initialized()

        total_count = await self.backend.count(start_time, end_time)
        last_entry = self._last_entry

        return {
            "total_entries": total_count,
            "last_sequence": last_entry.sequence_number if last_entry else 0,
            "last_entry_time": last_entry.timestamp.isoformat() if last_entry else None,
            "last_entry_hash": last_entry.entry_hash if last_entry else None,
        }


# Global instance
_audit_log: Optional[ImmutableAuditLog] = None
_lock = threading.Lock()


def get_audit_log() -> ImmutableAuditLog:
    """Get the global audit log instance."""
    global _audit_log

    if _audit_log is None:
        with _lock:
            if _audit_log is None:
                # Default to local file backend
                log_dir = os.environ.get(
                    "ARAGORA_AUDIT_LOG_DIR",
                    ".nomic/audit_logs",
                )
                signing_key = os.environ.get("ARAGORA_AUDIT_SIGNING_KEY")
                signing_key_bytes = signing_key.encode() if signing_key else None

                backend = LocalFileBackend(log_dir)
                _audit_log = ImmutableAuditLog(
                    backend=backend,
                    signing_key=signing_key_bytes,
                )

    return _audit_log


def init_audit_log(
    backend: AuditBackend = AuditBackend.LOCAL,
    signing_key: Optional[bytes] = None,
    **backend_kwargs: Any,
) -> ImmutableAuditLog:
    """
    Initialize the global audit log with specific configuration.

    Args:
        backend: Backend type to use
        signing_key: Optional signing key for HMAC signatures
        **backend_kwargs: Backend-specific configuration

    Returns:
        The initialized audit log
    """
    global _audit_log

    backend_impl: AuditLogBackend
    if backend == AuditBackend.LOCAL:
        log_dir = backend_kwargs.get("log_dir", ".nomic/audit_logs")
        backend_impl = LocalFileBackend(log_dir)
    elif backend == AuditBackend.S3_OBJECT_LOCK:
        backend_impl = S3ObjectLockBackend(
            bucket=backend_kwargs["bucket"],
            prefix=backend_kwargs.get("prefix", "audit-logs/"),
            region=backend_kwargs.get("region", "us-east-1"),
            retention_days=backend_kwargs.get("retention_days", 2555),
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    with _lock:
        _audit_log = ImmutableAuditLog(
            backend=backend_impl,
            signing_key=signing_key,
        )

    return _audit_log


# Convenience functions for common audit events
async def audit_finding_created(
    finding_id: str,
    actor: str,
    workspace_id: str,
    severity: str,
    category: str,
    **kwargs,
) -> AuditEntry:
    """Log finding creation."""
    return await get_audit_log().append(
        event_type="finding_created",
        actor=actor,
        resource_type="finding",
        resource_id=finding_id,
        action="create",
        workspace_id=workspace_id,
        details={"severity": severity, "category": category, **kwargs},
    )


async def audit_finding_updated(
    finding_id: str,
    actor: str,
    changes: dict[str, Any],
    **kwargs,
) -> AuditEntry:
    """Log finding update."""
    return await get_audit_log().append(
        event_type="finding_updated",
        actor=actor,
        resource_type="finding",
        resource_id=finding_id,
        action="update",
        details={"changes": changes, **kwargs},
    )


async def audit_document_uploaded(
    document_id: str,
    actor: str,
    workspace_id: str,
    filename: str,
    **kwargs,
) -> AuditEntry:
    """Log document upload."""
    return await get_audit_log().append(
        event_type="document_uploaded",
        actor=actor,
        resource_type="document",
        resource_id=document_id,
        action="upload",
        workspace_id=workspace_id,
        details={"filename": filename, **kwargs},
    )


async def audit_document_accessed(
    document_id: str,
    actor: str,
    access_type: str = "read",
    **kwargs,
) -> AuditEntry:
    """Log document access."""
    return await get_audit_log().append(
        event_type="document_accessed",
        actor=actor,
        resource_type="document",
        resource_id=document_id,
        action=access_type,
        details=kwargs,
    )


async def audit_session_started(
    session_id: str,
    actor: str,
    workspace_id: str,
    session_type: str,
    **kwargs,
) -> AuditEntry:
    """Log audit session start."""
    return await get_audit_log().append(
        event_type="session_started",
        actor=actor,
        resource_type="audit_session",
        resource_id=session_id,
        action="start",
        workspace_id=workspace_id,
        details={"session_type": session_type, **kwargs},
    )


async def audit_data_exported(
    export_id: str,
    actor: str,
    resource_type: str,
    resource_ids: list[str],
    format: str,
    **kwargs,
) -> AuditEntry:
    """Log data export."""
    return await get_audit_log().append(
        event_type="data_exported",
        actor=actor,
        resource_type="export",
        resource_id=export_id,
        action="export",
        details={
            "exported_type": resource_type,
            "exported_ids": resource_ids,
            "format": format,
            **kwargs,
        },
    )
