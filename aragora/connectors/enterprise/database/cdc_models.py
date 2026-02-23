"""
CDC Data Models and Enums.

Core data types for Change Data Capture (CDC) events, resume tokens,
and configuration shared across all CDC handlers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Enums
# =============================================================================


class ChangeOperation(str, Enum):
    """Type of database change operation."""

    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    REPLACE = "replace"  # MongoDB-specific
    UPSERT = "upsert"
    TRUNCATE = "truncate"
    SCHEMA_CHANGE = "schema_change"


class CDCSourceType(str, Enum):
    """Source database type for CDC events."""

    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    SNOWFLAKE = "snowflake"
    MYSQL = "mysql"
    SQLSERVER = "sqlserver"


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ChangeEvent:
    """
    Unified representation of a database change event.

    Works across PostgreSQL (LISTEN/NOTIFY, logical replication),
    MongoDB (change streams), and other databases.
    """

    # Event identification
    id: str  # Unique event ID
    source_type: CDCSourceType
    connector_id: str

    # Operation details
    operation: ChangeOperation
    timestamp: datetime

    # Location
    database: str
    schema: str | None = None  # PostgreSQL schema or MongoDB database
    table: str = ""  # Table or collection name

    # Key identification
    primary_key: dict[str, Any] | None = None  # Primary key values
    document_id: str | None = None  # MongoDB _id or row identifier

    # Data (for insert/update/replace)
    data: dict[str, Any] | None = None  # New data after change
    old_data: dict[str, Any] | None = None  # Old data before change (if available)

    # Change tracking
    fields_changed: list[str] = field(default_factory=list)

    # Resume support
    resume_token: str | None = None  # MongoDB resume token or LSN
    sequence_number: int | None = None  # For ordering

    # Metadata
    transaction_id: str | None = None
    user: str | None = None  # User who made the change
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate a unique event ID."""
        components = [
            self.source_type.value,
            self.database,
            self.table,
            self.operation.value,
            self.timestamp.isoformat() if self.timestamp else "",
            str(self.primary_key or self.document_id or ""),
        ]
        content = ":".join(components)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def is_data_change(self) -> bool:
        """Check if this event represents a data change (not schema change)."""
        return self.operation in (
            ChangeOperation.INSERT,
            ChangeOperation.UPDATE,
            ChangeOperation.DELETE,
            ChangeOperation.REPLACE,
            ChangeOperation.UPSERT,
        )

    @property
    def qualified_table(self) -> str:
        """Get fully qualified table name."""
        if self.schema:
            return f"{self.schema}.{self.table}"
        return self.table

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "source_type": self.source_type.value,
            "connector_id": self.connector_id,
            "operation": self.operation.value,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "database": self.database,
            "schema": self.schema,
            "table": self.table,
            "primary_key": self.primary_key,
            "document_id": self.document_id,
            "data": self.data,
            "old_data": self.old_data,
            "fields_changed": self.fields_changed,
            "resume_token": self.resume_token,
            "sequence_number": self.sequence_number,
            "transaction_id": self.transaction_id,
            "user": self.user,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChangeEvent:
        """Deserialize from dictionary."""
        timestamp = data.get("timestamp")
        if timestamp and isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            id=data.get("id", ""),
            source_type=CDCSourceType(data.get("source_type", "postgresql")),
            connector_id=data.get("connector_id", ""),
            operation=ChangeOperation(data.get("operation", "update")),
            timestamp=timestamp or datetime.now(timezone.utc),
            database=data.get("database", ""),
            schema=data.get("schema"),
            table=data.get("table", ""),
            primary_key=data.get("primary_key"),
            document_id=data.get("document_id"),
            data=data.get("data"),
            old_data=data.get("old_data"),
            fields_changed=data.get("fields_changed", []),
            resume_token=data.get("resume_token"),
            sequence_number=data.get("sequence_number"),
            transaction_id=data.get("transaction_id"),
            user=data.get("user"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_postgres_notify(
        cls,
        payload: str,
        channel: str,
        connector_id: str,
        database: str,
        schema: str = "public",
    ) -> ChangeEvent:
        """Create ChangeEvent from PostgreSQL NOTIFY payload."""
        try:
            data = json.loads(payload) if payload else {}
        except json.JSONDecodeError:
            data = {"raw": payload}

        operation_map = {
            "INSERT": ChangeOperation.INSERT,
            "UPDATE": ChangeOperation.UPDATE,
            "DELETE": ChangeOperation.DELETE,
            "TRUNCATE": ChangeOperation.TRUNCATE,
        }

        return cls(
            id="",
            source_type=CDCSourceType.POSTGRESQL,
            connector_id=connector_id,
            operation=operation_map.get(
                data.get("operation", "UPDATE").upper(), ChangeOperation.UPDATE
            ),
            timestamp=datetime.now(timezone.utc),
            database=database,
            schema=schema,
            table=data.get("table", channel),
            primary_key=data.get("primary_key"),
            data=data.get("new_data") or data.get("data"),
            old_data=data.get("old_data"),
            fields_changed=data.get("changed_columns", []),
            transaction_id=data.get("xid"),
            user=data.get("user"),
            metadata={"channel": channel, "raw_payload": payload},
        )

    @classmethod
    def from_mongodb_change(
        cls,
        change: dict[str, Any],
        connector_id: str,
    ) -> ChangeEvent:
        """Create ChangeEvent from MongoDB change stream document."""
        operation_map = {
            "insert": ChangeOperation.INSERT,
            "update": ChangeOperation.UPDATE,
            "replace": ChangeOperation.REPLACE,
            "delete": ChangeOperation.DELETE,
        }

        ns = change.get("ns", {})
        document_key = change.get("documentKey", {})
        full_document = change.get("fullDocument")
        update_description = change.get("updateDescription", {})

        # Extract document ID
        doc_id = document_key.get("_id")
        if doc_id and hasattr(doc_id, "__str__"):
            doc_id = str(doc_id)

        # Get changed fields from update description
        fields_changed = []
        if update_description:
            updated = update_description.get("updatedFields", {})
            removed = update_description.get("removedFields", [])
            fields_changed = list(updated.keys()) + removed

        # Get resume token
        resume_token = change.get("_id")
        if resume_token:
            resume_token = json.dumps(resume_token, default=str)

        return cls(
            id="",
            source_type=CDCSourceType.MONGODB,
            connector_id=connector_id,
            operation=operation_map.get(
                change.get("operationType", "update"), ChangeOperation.UPDATE
            ),
            timestamp=change.get("clusterTime", datetime.now(timezone.utc)),
            database=ns.get("db", ""),
            table=ns.get("coll", ""),
            document_id=doc_id,
            data=full_document,
            old_data=change.get("fullDocumentBeforeChange"),
            fields_changed=fields_changed,
            resume_token=resume_token,
            transaction_id=str(change.get("txnNumber")) if change.get("txnNumber") else None,
            metadata={"operation_type": change.get("operationType")},
        )


# =============================================================================
# Resume Token Management
# =============================================================================


@dataclass
class ResumeToken:
    """
    Persistent resume token for CDC streams.

    Enables resuming change streams from the last processed position
    after connector restarts.
    """

    connector_id: str
    source_type: CDCSourceType
    token: str  # Opaque token (MongoDB resume token, PostgreSQL LSN, etc.)
    timestamp: datetime
    sequence_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "connector_id": self.connector_id,
            "source_type": self.source_type.value,
            "token": self.token,
            "timestamp": self.timestamp.isoformat(),
            "sequence_number": self.sequence_number,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResumeToken:
        timestamp = data.get("timestamp")
        if timestamp and isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            connector_id=data.get("connector_id", ""),
            source_type=CDCSourceType(data.get("source_type", "postgresql")),
            token=data.get("token", ""),
            timestamp=timestamp or datetime.now(timezone.utc),
            sequence_number=data.get("sequence_number"),
            metadata=data.get("metadata", {}),
        )


class ResumeTokenStore:
    """
    Persistent storage for CDC resume tokens.

    Stores tokens in a JSON file for simple file-based persistence.
    Can be extended for database-backed storage.
    """

    def __init__(self, storage_path: Path | None = None):
        if storage_path:
            self.storage_path = storage_path
        else:
            from aragora.persistence.db_config import get_nomic_dir

            self.storage_path = get_nomic_dir() / "cdc_resume_tokens.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._tokens: dict[str, ResumeToken] = {}
        self._load()

    def _load(self) -> None:
        """Load tokens from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                    for key, value in data.items():
                        self._tokens[key] = ResumeToken.from_dict(value)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to load resume tokens: %s", e)
                self._tokens = {}

    def _save(self) -> None:
        """Save tokens to storage."""
        try:
            with open(self.storage_path, "w") as f:
                data = {key: token.to_dict() for key, token in self._tokens.items()}
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.error("Failed to save resume tokens: %s", e)

    def get(self, connector_id: str) -> ResumeToken | None:
        """Get resume token for a connector."""
        return self._tokens.get(connector_id)

    def save(self, token: ResumeToken) -> None:
        """Save a resume token."""
        self._tokens[token.connector_id] = token
        self._save()

    def delete(self, connector_id: str) -> None:
        """Delete a resume token."""
        if connector_id in self._tokens:
            del self._tokens[connector_id]
            self._save()

    def clear_all(self) -> None:
        """Clear all resume tokens."""
        self._tokens = {}
        self._save()


# =============================================================================
# CDC Configuration
# =============================================================================


@dataclass
class CDCConfig:
    """
    Configuration for CDC handlers.

    Provides common settings for retry behavior, reconnection,
    and event filtering.
    """

    # Reconnection settings
    max_retries: int = 5
    retry_delay_seconds: float = 1.0
    max_retry_delay_seconds: float = 60.0
    retry_backoff_multiplier: float = 2.0

    # Event filtering
    include_tables: list[str] | None = None  # None = all tables
    exclude_tables: list[str] | None = None
    include_operations: list[ChangeOperation] | None = None  # None = all operations

    # Processing settings
    batch_size: int = 100
    commit_interval_seconds: float = 5.0

    # Health monitoring
    health_check_interval_seconds: float = 30.0
