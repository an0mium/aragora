"""
Audit Log Storage Backends.

Provides abstract base class and implementations for:
- LocalFileBackend: Append-only file (dev/testing)
- PostgreSQLAuditBackend: PostgreSQL with indexed queries (production)
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

                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

        return count

    async def save_anchor(self, anchor: DailyAnchor) -> None:
        """Save daily anchor."""
        anchors = {}
        if self.anchors_file.exists():
            try:
                anchors = json.loads(self.anchors_file.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not load anchors file, starting fresh: {e}")

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
            except (KeyError, json.JSONDecodeError, ConnectionError) as e:
                logger.debug(f"Cache miss for sequence {sequence_number}: {e}")
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


class PostgreSQLAuditBackend(AuditLogBackend):
    """
    PostgreSQL backend for immutable audit logs.

    Uses a dedicated table with JSONB columns for efficient querying.
    Supports hash chain verification and sequence-based lookups.
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        table_prefix: str = "immutable_",
    ):
        """
        Initialize PostgreSQL audit backend.

        Args:
            database_url: PostgreSQL connection URL. If not provided,
                         uses DATABASE_URL or ARAGORA_DATABASE_URL env var.
            table_prefix: Prefix for table names (default: "immutable_")
        """
        import os

        self.database_url = (
            database_url or os.environ.get("DATABASE_URL") or os.environ.get("ARAGORA_DATABASE_URL")
        )
        if not self.database_url:
            raise ValueError("PostgreSQL backend requires DATABASE_URL or ARAGORA_DATABASE_URL")

        self.table_prefix = table_prefix
        self.entries_table = f"{table_prefix}audit_entries"
        self.anchors_table = f"{table_prefix}daily_anchors"

        self._pool: Any = None
        self._init_db()

    def _get_pool(self) -> Any:
        """Get or create connection pool."""
        if self._pool is None:
            try:
                from psycopg2 import pool as pg_pool

                self._pool = pg_pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=10,
                    dsn=self.database_url,
                )
            except ImportError:
                raise ImportError(
                    "psycopg2 required for PostgreSQL backend: pip install psycopg2-binary"
                )
        return self._pool

    def _init_db(self) -> None:
        """Initialize database schema."""
        pool = self._get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                # Create entries table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.entries_table} (
                        id TEXT PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        sequence_number BIGINT NOT NULL UNIQUE,
                        previous_hash TEXT NOT NULL,
                        entry_hash TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        actor TEXT NOT NULL,
                        actor_type TEXT NOT NULL DEFAULT 'user',
                        resource_type TEXT NOT NULL,
                        resource_id TEXT NOT NULL,
                        action TEXT NOT NULL,
                        details JSONB DEFAULT '{{}}',
                        correlation_id TEXT,
                        workspace_id TEXT,
                        ip_address TEXT,
                        user_agent TEXT,
                        signature TEXT
                    )
                """)

                # Create indexes for common queries
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}audit_timestamp
                    ON {self.entries_table}(timestamp DESC)
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}audit_event_type
                    ON {self.entries_table}(event_type)
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}audit_actor
                    ON {self.entries_table}(actor)
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}audit_resource
                    ON {self.entries_table}(resource_type, resource_id)
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}audit_workspace
                    ON {self.entries_table}(workspace_id) WHERE workspace_id IS NOT NULL
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}audit_sequence
                    ON {self.entries_table}(sequence_number)
                """)

                # Create anchors table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.anchors_table} (
                        date TEXT PRIMARY KEY,
                        first_sequence BIGINT NOT NULL,
                        last_sequence BIGINT NOT NULL,
                        entry_count INTEGER NOT NULL,
                        merkle_root TEXT NOT NULL,
                        chain_hash TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL
                    )
                """)

                conn.commit()
                logger.info(f"PostgreSQL audit tables initialized: {self.entries_table}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to initialize PostgreSQL audit tables: {e}")
            raise
        finally:
            pool.putconn(conn)

    async def append(self, entry: AuditEntry) -> None:
        """Append entry to PostgreSQL."""
        pool = self._get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.entries_table}
                    (id, timestamp, sequence_number, previous_hash, entry_hash,
                     event_type, actor, actor_type, resource_type, resource_id,
                     action, details, correlation_id, workspace_id, ip_address,
                     user_agent, signature)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        entry.id,
                        entry.timestamp,
                        entry.sequence_number,
                        entry.previous_hash,
                        entry.entry_hash,
                        entry.event_type,
                        entry.actor,
                        entry.actor_type,
                        entry.resource_type,
                        entry.resource_id,
                        entry.action,
                        json.dumps(entry.details),
                        entry.correlation_id,
                        entry.workspace_id,
                        entry.ip_address,
                        entry.user_agent,
                        entry.signature,
                    ),
                )
                conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to append audit entry to PostgreSQL: {e}")
            raise
        finally:
            pool.putconn(conn)

    def _row_to_entry(self, row: tuple) -> AuditEntry:
        """Convert database row to AuditEntry."""
        return AuditEntry(
            id=row[0],
            timestamp=row[1],
            sequence_number=row[2],
            previous_hash=row[3],
            entry_hash=row[4],
            event_type=row[5],
            actor=row[6],
            actor_type=row[7],
            resource_type=row[8],
            resource_id=row[9],
            action=row[10],
            details=row[11] if isinstance(row[11], dict) else json.loads(row[11] or "{}"),
            correlation_id=row[12],
            workspace_id=row[13],
            ip_address=row[14],
            user_agent=row[15],
            signature=row[16],
        )

    async def get_entry(self, entry_id: str) -> Optional[AuditEntry]:
        """Get entry by ID."""
        pool = self._get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, timestamp, sequence_number, previous_hash, entry_hash,
                           event_type, actor, actor_type, resource_type, resource_id,
                           action, details, correlation_id, workspace_id, ip_address,
                           user_agent, signature
                    FROM {self.entries_table}
                    WHERE id = %s
                    """,
                    (entry_id,),
                )
                row = cur.fetchone()
                if row:
                    return self._row_to_entry(row)
                return None
        finally:
            pool.putconn(conn)

    async def get_by_sequence(self, sequence_number: int) -> Optional[AuditEntry]:
        """Get entry by sequence number."""
        pool = self._get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, timestamp, sequence_number, previous_hash, entry_hash,
                           event_type, actor, actor_type, resource_type, resource_id,
                           action, details, correlation_id, workspace_id, ip_address,
                           user_agent, signature
                    FROM {self.entries_table}
                    WHERE sequence_number = %s
                    """,
                    (sequence_number,),
                )
                row = cur.fetchone()
                if row:
                    return self._row_to_entry(row)
                return None
        finally:
            pool.putconn(conn)

    async def get_last_entry(self) -> Optional[AuditEntry]:
        """Get most recent entry."""
        pool = self._get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, timestamp, sequence_number, previous_hash, entry_hash,
                           event_type, actor, actor_type, resource_type, resource_id,
                           action, details, correlation_id, workspace_id, ip_address,
                           user_agent, signature
                    FROM {self.entries_table}
                    ORDER BY sequence_number DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
                if row:
                    return self._row_to_entry(row)
                return None
        finally:
            pool.putconn(conn)

    async def get_entries_range(
        self,
        start_sequence: int,
        end_sequence: int,
    ) -> list[AuditEntry]:
        """Get entries in sequence range."""
        pool = self._get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, timestamp, sequence_number, previous_hash, entry_hash,
                           event_type, actor, actor_type, resource_type, resource_id,
                           action, details, correlation_id, workspace_id, ip_address,
                           user_agent, signature
                    FROM {self.entries_table}
                    WHERE sequence_number BETWEEN %s AND %s
                    ORDER BY sequence_number
                    """,
                    (start_sequence, end_sequence),
                )
                return [self._row_to_entry(row) for row in cur.fetchall()]
        finally:
            pool.putconn(conn)

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
        conditions = ["TRUE"]
        params: list[Any] = []

        if start_time:
            conditions.append("timestamp >= %s")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= %s")
            params.append(end_time)
        if event_types:
            conditions.append("event_type = ANY(%s)")
            params.append(event_types)
        if actors:
            conditions.append("actor = ANY(%s)")
            params.append(actors)
        if resource_types:
            conditions.append("resource_type = ANY(%s)")
            params.append(resource_types)
        if resource_ids:
            conditions.append("resource_id = ANY(%s)")
            params.append(resource_ids)
        if workspace_id:
            conditions.append("workspace_id = %s")
            params.append(workspace_id)

        where_clause = " AND ".join(conditions)
        params.extend([limit, offset])

        pool = self._get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, timestamp, sequence_number, previous_hash, entry_hash,
                           event_type, actor, actor_type, resource_type, resource_id,
                           action, details, correlation_id, workspace_id, ip_address,
                           user_agent, signature
                    FROM {self.entries_table}
                    WHERE {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT %s OFFSET %s
                    """,
                    tuple(params),
                )
                return [self._row_to_entry(row) for row in cur.fetchall()]
        finally:
            pool.putconn(conn)

    async def count(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[list[str]] = None,
    ) -> int:
        """Count entries matching filters."""
        conditions = ["TRUE"]
        params: list[Any] = []

        if start_time:
            conditions.append("timestamp >= %s")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= %s")
            params.append(end_time)
        if event_types:
            conditions.append("event_type = ANY(%s)")
            params.append(event_types)

        where_clause = " AND ".join(conditions)

        pool = self._get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT COUNT(*) FROM {self.entries_table} WHERE {where_clause}",
                    tuple(params),
                )
                result = cur.fetchone()
                return result[0] if result else 0
        finally:
            pool.putconn(conn)

    async def save_anchor(self, anchor: DailyAnchor) -> None:
        """Save daily anchor."""
        pool = self._get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.anchors_table}
                    (date, first_sequence, last_sequence, entry_count,
                     merkle_root, chain_hash, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date) DO UPDATE SET
                        first_sequence = EXCLUDED.first_sequence,
                        last_sequence = EXCLUDED.last_sequence,
                        entry_count = EXCLUDED.entry_count,
                        merkle_root = EXCLUDED.merkle_root,
                        chain_hash = EXCLUDED.chain_hash,
                        created_at = EXCLUDED.created_at
                    """,
                    (
                        anchor.date,
                        anchor.first_sequence,
                        anchor.last_sequence,
                        anchor.entry_count,
                        anchor.merkle_root,
                        anchor.chain_hash,
                        anchor.created_at,
                    ),
                )
                conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to save anchor to PostgreSQL: {e}")
            raise
        finally:
            pool.putconn(conn)

    async def get_anchor(self, date: str) -> Optional[DailyAnchor]:
        """Get anchor for date."""
        pool = self._get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT date, first_sequence, last_sequence, entry_count,
                           merkle_root, chain_hash, created_at
                    FROM {self.anchors_table}
                    WHERE date = %s
                    """,
                    (date,),
                )
                row = cur.fetchone()
                if row:
                    return DailyAnchor(
                        date=row[0],
                        first_sequence=row[1],
                        last_sequence=row[2],
                        entry_count=row[3],
                        merkle_root=row[4],
                        chain_hash=row[5],
                        created_at=row[6],
                    )
                return None
        finally:
            pool.putconn(conn)

    def close(self) -> None:
        """Close connection pool."""
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None


def create_audit_backend(
    backend_type: str = "local",
    **kwargs: Any,
) -> AuditLogBackend:
    """
    Factory function to create audit log backends.

    Args:
        backend_type: Type of backend ("local", "postgresql", "s3_object_lock")
        **kwargs: Backend-specific configuration

    Returns:
        Configured AuditLogBackend instance

    Examples:
        # Local file backend (default)
        backend = create_audit_backend("local", log_dir="/var/log/aragora/audit")

        # PostgreSQL backend
        backend = create_audit_backend(
            "postgresql",
            database_url="postgresql://user:pass@host/db"
        )

        # S3 Object Lock backend
        backend = create_audit_backend(
            "s3_object_lock",
            bucket="my-audit-bucket",
            region="us-east-1"
        )
    """
    import os

    backend_type = backend_type.lower()

    if backend_type == "local":
        log_dir = kwargs.get("log_dir", os.environ.get("ARAGORA_AUDIT_LOG_DIR", "./audit_logs"))
        return LocalFileBackend(log_dir)

    elif backend_type == "postgresql":
        database_url = kwargs.get("database_url")
        table_prefix = kwargs.get("table_prefix", "immutable_")
        return PostgreSQLAuditBackend(
            database_url=database_url,
            table_prefix=table_prefix,
        )

    elif backend_type == "s3_object_lock":
        bucket = kwargs.get("bucket")
        if not bucket:
            bucket = os.environ.get("ARAGORA_AUDIT_S3_BUCKET")
        if not bucket:
            raise ValueError(
                "S3 backend requires 'bucket' parameter or ARAGORA_AUDIT_S3_BUCKET env var"
            )
        return S3ObjectLockBackend(
            bucket=bucket,
            prefix=kwargs.get("prefix", "audit-logs/"),
            region=kwargs.get("region", os.environ.get("AWS_REGION", "us-east-1")),
            retention_days=kwargs.get("retention_days", 2555),
        )

    else:
        raise ValueError(
            f"Unknown backend type: {backend_type}. Supported: local, postgresql, s3_object_lock"
        )
