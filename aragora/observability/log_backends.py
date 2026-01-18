"""
Audit Log Storage Backends.

Provides abstract base class and implementations for:
- LocalFileBackend: Append-only file (dev/testing)
- S3ObjectLockBackend: S3 with Object Lock (WORM compliance)
"""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from aragora.observability.log_types import AuditEntry, DailyAnchor

logger = logging.getLogger(__name__)


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
