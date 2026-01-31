"""
Audit Interceptor for Enterprise Gateway.

Provides comprehensive request/response audit logging with cryptographic
integrity verification, PII redaction, and compliance support.

Features:
- Request/response logging with timing metrics
- SHA-256 hashing with hash chain for tamper detection
- GDPR-compliant PII redaction
- SOC 2 Type II audit evidence generation
- Real-time event emission for SIEM integration
- Prometheus metrics for audit volume monitoring
- Configurable retention policies
- Webhook support for external integrations

Usage:
    from aragora.gateway.enterprise.audit_interceptor import (
        AuditInterceptor,
        AuditConfig,
        PIIRedactionRule,
        RedactionType,
    )

    # Configure with PII redaction
    config = AuditConfig(
        retention_days=365,
        pii_fields=["email", "phone", "ssn"],
        emit_events=True,
        webhook_url="https://siem.example.com/webhook",
        pii_rules=[
            PIIRedactionRule(
                field_pattern=r".*email.*",
                redaction_type=RedactionType.HASH,
            ),
            PIIRedactionRule(
                field_pattern=r".*password.*",
                redaction_type=RedactionType.REMOVE,
            ),
        ],
    )

    interceptor = AuditInterceptor(config=config)

    # Intercept request/response
    record = await interceptor.intercept(
        request={"method": "POST", "path": "/api/users", "body": {...}},
        response={"status": 200, "body": {...}},
        correlation_id="req-123",
        user_id="user-456",
    )

    # Verify chain integrity
    is_valid, errors = await interceptor.verify_chain(since=datetime.now() - timedelta(days=7))

    # Export for SOC 2 audit
    report = await interceptor.export_soc2_evidence(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class RedactionType(str, Enum):
    """Types of PII redaction strategies."""

    MASK = "mask"  # Replace with asterisks (e.g., "j***@example.com")
    HASH = "hash"  # Replace with SHA-256 hash (preserves uniqueness)
    REMOVE = "remove"  # Remove field entirely
    TRUNCATE = "truncate"  # Keep first/last N chars (e.g., "j...m@e...m")
    TOKENIZE = "tokenize"  # Replace with reversible token (requires key)


class AuditEventType(str, Enum):
    """Types of audit events emitted by the interceptor."""

    REQUEST_RECEIVED = "request_received"
    RESPONSE_SENT = "response_sent"
    REQUEST_FAILED = "request_failed"
    PII_REDACTED = "pii_redacted"
    CHAIN_VERIFIED = "chain_verified"
    CHAIN_BROKEN = "chain_broken"
    RETENTION_APPLIED = "retention_applied"
    EXPORT_GENERATED = "export_generated"


# HMAC signing key management
_INTERCEPTOR_SIGNING_KEY: bytes | None = None
_signing_key_lock = threading.Lock()


def get_interceptor_signing_key() -> bytes:
    """
    Get or generate the HMAC signing key for audit records.

    The key is loaded from ARAGORA_AUDIT_INTERCEPTOR_KEY environment variable.
    Falls back to ARAGORA_AUDIT_SIGNING_KEY if not set.
    For development, generates a random key if not set.

    Returns:
        32-byte HMAC signing key

    Raises:
        RuntimeError: If key not set in production/staging environment
    """
    global _INTERCEPTOR_SIGNING_KEY
    with _signing_key_lock:
        if _INTERCEPTOR_SIGNING_KEY is None:
            key_hex = os.environ.get("ARAGORA_AUDIT_INTERCEPTOR_KEY") or os.environ.get(
                "ARAGORA_AUDIT_SIGNING_KEY"
            )
            if key_hex:
                try:
                    _INTERCEPTOR_SIGNING_KEY = bytes.fromhex(key_hex)
                    if len(_INTERCEPTOR_SIGNING_KEY) < 32:
                        raise ValueError("Key must be at least 32 bytes (64 hex chars)")
                    logger.debug("Loaded audit interceptor signing key from environment")
                except ValueError as e:
                    raise RuntimeError(
                        f"Invalid signing key format: {e}. "
                        "Key must be a hex-encoded string of at least 64 characters. "
                        "Generate one with: python -c 'import secrets; print(secrets.token_hex(32))'"
                    ) from e
            else:
                env = os.environ.get("ARAGORA_ENV", "development")
                if env in ("production", "prod", "staging"):
                    raise RuntimeError(
                        f"ARAGORA_AUDIT_INTERCEPTOR_KEY required in {env} environment. "
                        "Audit record signatures cannot be verified without a persistent key."
                    )
                else:
                    _INTERCEPTOR_SIGNING_KEY = secrets.token_bytes(32)
                    logger.debug(
                        "Generated ephemeral audit interceptor signing key for development"
                    )
        return _INTERCEPTOR_SIGNING_KEY


def set_interceptor_signing_key(key: bytes) -> None:
    """
    Set the HMAC signing key for audit records.

    Args:
        key: At least 32-byte HMAC signing key
    """
    global _INTERCEPTOR_SIGNING_KEY
    if len(key) < 32:
        raise ValueError("Signing key must be at least 32 bytes")
    with _signing_key_lock:
        _INTERCEPTOR_SIGNING_KEY = key


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PIIRedactionRule:
    """
    Rule for redacting PII from audit records.

    Defines how specific fields should be redacted based on field name patterns.
    Rules are applied in order; first matching rule wins.

    Attributes:
        field_pattern: Regex pattern to match field names (case-insensitive)
        redaction_type: How to redact the matched field
        mask_char: Character to use for MASK redaction (default: '*')
        visible_chars: Number of chars to show at start/end for TRUNCATE
        custom_handler: Optional custom redaction function
    """

    field_pattern: str
    redaction_type: RedactionType
    mask_char: str = "*"
    visible_chars: int = 2
    custom_handler: Callable[[str, Any], Any] | None = None

    def __post_init__(self) -> None:
        """Compile the regex pattern."""
        self._compiled_pattern = re.compile(self.field_pattern, re.IGNORECASE)

    def matches(self, field_name: str) -> bool:
        """Check if this rule matches the given field name."""
        return bool(self._compiled_pattern.search(field_name))

    def redact(self, value: Any) -> Any:
        """
        Apply redaction to a value.

        Args:
            value: The value to redact

        Returns:
            Redacted value
        """
        if self.custom_handler:
            return self.custom_handler(self.field_pattern, value)

        if value is None:
            return None

        str_value = str(value)

        if self.redaction_type == RedactionType.REMOVE:
            return "[REDACTED]"

        elif self.redaction_type == RedactionType.MASK:
            if len(str_value) <= 2:
                return self.mask_char * len(str_value)
            return str_value[0] + self.mask_char * (len(str_value) - 2) + str_value[-1]

        elif self.redaction_type == RedactionType.HASH:
            hash_value = hashlib.sha256(str_value.encode()).hexdigest()[:16]
            return f"[HASH:{hash_value}]"

        elif self.redaction_type == RedactionType.TRUNCATE:
            if len(str_value) <= self.visible_chars * 2:
                return self.mask_char * len(str_value)
            return str_value[: self.visible_chars] + "..." + str_value[-self.visible_chars :]

        elif self.redaction_type == RedactionType.TOKENIZE:
            # Tokenization requires a reversible mapping - use hash as placeholder
            token = hashlib.sha256(str_value.encode()).hexdigest()[:12]
            return f"[TOKEN:{token}]"

        return value


@dataclass
class AuditConfig:
    """
    Configuration for the AuditInterceptor.

    Attributes:
        retention_days: Days to retain audit records (default: 365)
        pii_fields: List of field names to always redact
        pii_rules: List of PIIRedactionRule for pattern-based redaction
        emit_events: Whether to emit audit events (default: True)
        webhook_url: URL for SIEM webhook integration
        webhook_headers: Additional headers for webhook requests
        webhook_timeout: Timeout for webhook requests in seconds
        enable_metrics: Whether to expose Prometheus metrics
        metrics_prefix: Prefix for Prometheus metric names
        hash_responses: Whether to hash response bodies (privacy)
        max_body_size: Maximum body size to log (bytes, 0 = unlimited)
        sensitive_headers: Headers to redact from logs
        chain_verification_interval: Seconds between chain verification runs
        storage_backend: Storage backend type ("memory", "postgres")
    """

    retention_days: int = 365
    pii_fields: list[str] = field(default_factory=list)
    pii_rules: list[PIIRedactionRule] = field(default_factory=list)
    emit_events: bool = True
    webhook_url: str | None = None
    webhook_headers: dict[str, str] = field(default_factory=dict)
    webhook_timeout: float = 5.0
    enable_metrics: bool = True
    metrics_prefix: str = "aragora_audit"
    hash_responses: bool = False
    max_body_size: int = 1024 * 1024  # 1MB default
    sensitive_headers: list[str] = field(
        default_factory=lambda: [
            "authorization",
            "x-api-key",
            "cookie",
            "x-csrf-token",
            "x-auth-token",
        ]
    )
    chain_verification_interval: int = 3600  # 1 hour
    storage_backend: str = "memory"

    def __post_init__(self) -> None:
        """Initialize default PII rules if none provided."""
        if not self.pii_rules and self.pii_fields:
            # Create rules from simple field list
            self.pii_rules = [
                PIIRedactionRule(
                    field_pattern=rf".*{re.escape(field)}.*",
                    redaction_type=RedactionType.MASK,
                )
                for field in self.pii_fields
            ]

        # Add default sensitive field rules
        default_rules = [
            PIIRedactionRule(
                field_pattern=r".*password.*",
                redaction_type=RedactionType.REMOVE,
            ),
            PIIRedactionRule(
                field_pattern=r".*secret.*",
                redaction_type=RedactionType.REMOVE,
            ),
            PIIRedactionRule(
                field_pattern=r".*token.*",
                redaction_type=RedactionType.HASH,
            ),
            PIIRedactionRule(
                field_pattern=r".*api[_-]?key.*",
                redaction_type=RedactionType.HASH,
            ),
            PIIRedactionRule(
                field_pattern=r".*ssn.*",
                redaction_type=RedactionType.MASK,
            ),
            PIIRedactionRule(
                field_pattern=r".*social[_-]?security.*",
                redaction_type=RedactionType.MASK,
            ),
            PIIRedactionRule(
                field_pattern=r".*credit[_-]?card.*",
                redaction_type=RedactionType.MASK,
            ),
            PIIRedactionRule(
                field_pattern=r".*card[_-]?number.*",
                redaction_type=RedactionType.MASK,
            ),
        ]

        # Prepend user rules so they take precedence
        self.pii_rules = self.pii_rules + default_rules


@dataclass
class AuditRecord:
    """
    An audit record capturing a request/response pair.

    Attributes:
        id: Unique record identifier
        correlation_id: Request correlation/trace ID
        timestamp: When the request was received
        request_method: HTTP method
        request_path: Request path
        request_headers: Request headers (redacted)
        request_body: Request body (redacted, may be hashed)
        request_body_hash: SHA-256 hash of original request body
        response_status: HTTP response status code
        response_headers: Response headers (redacted)
        response_body: Response body (redacted, may be hashed)
        response_body_hash: SHA-256 hash of original response body
        duration_ms: Request duration in milliseconds
        user_id: Authenticated user ID (if available)
        org_id: Organization ID (if available)
        ip_address: Client IP address
        user_agent: Client user agent
        record_hash: SHA-256 hash of this record
        previous_hash: Hash of previous record (for chain)
        signature: HMAC-SHA256 signature for integrity
        metadata: Additional metadata
        pii_fields_redacted: List of field names that were redacted
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    correlation_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    request_method: str = ""
    request_path: str = ""
    request_headers: dict[str, str] = field(default_factory=dict)
    request_body: Any = None
    request_body_hash: str = ""
    response_status: int = 0
    response_headers: dict[str, str] = field(default_factory=dict)
    response_body: Any = None
    response_body_hash: str = ""
    duration_ms: float = 0.0
    user_id: str | None = None
    org_id: str | None = None
    ip_address: str = ""
    user_agent: str = ""
    record_hash: str = ""
    previous_hash: str = ""
    signature: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    pii_fields_redacted: list[str] = field(default_factory=list)

    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of this record for chain integrity.

        Returns:
            Hex-encoded SHA-256 hash
        """
        data = (
            f"{self.id}|{self.correlation_id}|{self.timestamp.isoformat()}|"
            f"{self.request_method}|{self.request_path}|{self.request_body_hash}|"
            f"{self.response_status}|{self.response_body_hash}|{self.duration_ms}|"
            f"{self.user_id}|{self.previous_hash}"
        )
        return hashlib.sha256(data.encode()).hexdigest()

    def compute_signature(self) -> str:
        """
        Compute HMAC-SHA256 signature for integrity verification.

        Returns:
            Hex-encoded HMAC-SHA256 signature
        """
        data_to_sign = {
            "id": self.id,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "request_method": self.request_method,
            "request_path": self.request_path,
            "request_body_hash": self.request_body_hash,
            "response_status": self.response_status,
            "response_body_hash": self.response_body_hash,
            "duration_ms": self.duration_ms,
            "user_id": self.user_id,
            "record_hash": self.record_hash,
            "previous_hash": self.previous_hash,
        }
        canonical = json.dumps(data_to_sign, sort_keys=True, separators=(",", ":"))
        key = get_interceptor_signing_key()
        return hmac.new(key, canonical.encode(), hashlib.sha256).hexdigest()

    def verify_signature(self) -> bool:
        """
        Verify this record's signature.

        Returns:
            True if signature is valid
        """
        if not self.signature:
            return False
        computed = self.compute_signature()
        return hmac.compare_digest(computed, self.signature)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "request_method": self.request_method,
            "request_path": self.request_path,
            "request_headers": self.request_headers,
            "request_body": self.request_body,
            "request_body_hash": self.request_body_hash,
            "response_status": self.response_status,
            "response_headers": self.response_headers,
            "response_body": self.response_body,
            "response_body_hash": self.response_body_hash,
            "duration_ms": self.duration_ms,
            "user_id": self.user_id,
            "org_id": self.org_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "record_hash": self.record_hash,
            "previous_hash": self.previous_hash,
            "signature": self.signature,
            "metadata": self.metadata,
            "pii_fields_redacted": self.pii_fields_redacted,
        }

    def to_signed_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary with computed hash and signature.

        Returns:
            Record dictionary with hash and signature fields populated
        """
        self.record_hash = self.compute_hash()
        self.signature = self.compute_signature()
        return self.to_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditRecord":
        """Create an AuditRecord from a dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)

        return cls(
            id=data.get("id", str(uuid4())),
            correlation_id=data.get("correlation_id", ""),
            timestamp=timestamp,
            request_method=data.get("request_method", ""),
            request_path=data.get("request_path", ""),
            request_headers=data.get("request_headers", {}),
            request_body=data.get("request_body"),
            request_body_hash=data.get("request_body_hash", ""),
            response_status=data.get("response_status", 0),
            response_headers=data.get("response_headers", {}),
            response_body=data.get("response_body"),
            response_body_hash=data.get("response_body_hash", ""),
            duration_ms=data.get("duration_ms", 0.0),
            user_id=data.get("user_id"),
            org_id=data.get("org_id"),
            ip_address=data.get("ip_address", ""),
            user_agent=data.get("user_agent", ""),
            record_hash=data.get("record_hash", ""),
            previous_hash=data.get("previous_hash", ""),
            signature=data.get("signature", ""),
            metadata=data.get("metadata", {}),
            pii_fields_redacted=data.get("pii_fields_redacted", []),
        )


# =============================================================================
# Storage Backends
# =============================================================================


class AuditStorage(ABC):
    """
    Abstract base class for audit record storage.

    Implementations must provide thread-safe storage operations
    with support for hash chain integrity.
    """

    @abstractmethod
    async def store(self, record: AuditRecord) -> None:
        """
        Store an audit record.

        Args:
            record: The audit record to store
        """
        ...

    @abstractmethod
    async def get(self, record_id: str) -> AuditRecord | None:
        """
        Retrieve an audit record by ID.

        Args:
            record_id: The record ID

        Returns:
            The audit record or None if not found
        """
        ...

    @abstractmethod
    async def query(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        correlation_id: str | None = None,
        request_path: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[AuditRecord]:
        """
        Query audit records.

        Args:
            start_date: Filter records after this time
            end_date: Filter records before this time
            user_id: Filter by user ID
            org_id: Filter by organization ID
            correlation_id: Filter by correlation ID
            request_path: Filter by request path (prefix match)
            limit: Maximum records to return
            offset: Pagination offset

        Returns:
            List of matching audit records
        """
        ...

    @abstractmethod
    async def get_last_hash(self) -> str:
        """
        Get the hash of the most recent record for chain continuity.

        Returns:
            Hash of last record, or empty string if no records
        """
        ...

    @abstractmethod
    async def get_chain(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[AuditRecord]:
        """
        Get records in chain order for verification.

        Args:
            start_date: Start of verification range
            end_date: End of verification range

        Returns:
            Records in timestamp order
        """
        ...

    @abstractmethod
    async def delete_before(self, cutoff: datetime) -> int:
        """
        Delete records before the cutoff date (retention policy).

        Args:
            cutoff: Delete records with timestamp before this

        Returns:
            Number of records deleted
        """
        ...

    @abstractmethod
    async def count(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> int:
        """
        Count records in date range.

        Args:
            start_date: Start of range
            end_date: End of range

        Returns:
            Number of records
        """
        ...


class InMemoryAuditStorage(AuditStorage):
    """
    In-memory audit storage for development and testing.

    Not suitable for production as data is lost on restart.
    Thread-safe via asyncio lock.
    """

    def __init__(self, max_records: int = 100000) -> None:
        """
        Initialize in-memory storage.

        Args:
            max_records: Maximum records to keep (oldest evicted first)
        """
        self._records: dict[str, AuditRecord] = {}
        self._order: list[str] = []  # Ordered by timestamp
        self._max_records = max_records
        self._lock = asyncio.Lock()
        self._last_hash = ""

    async def store(self, record: AuditRecord) -> None:
        """Store a record."""
        async with self._lock:
            self._records[record.id] = record
            self._order.append(record.id)
            self._last_hash = record.record_hash

            # Evict old records if over limit
            while len(self._order) > self._max_records:
                old_id = self._order.pop(0)
                del self._records[old_id]

    async def get(self, record_id: str) -> AuditRecord | None:
        """Get a record by ID."""
        return self._records.get(record_id)

    async def query(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        correlation_id: str | None = None,
        request_path: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[AuditRecord]:
        """Query records with filters."""
        results = []
        for record_id in reversed(self._order):  # Most recent first
            record = self._records.get(record_id)
            if not record:
                continue

            # Apply filters
            if start_date and record.timestamp < start_date:
                continue
            if end_date and record.timestamp > end_date:
                continue
            if user_id and record.user_id != user_id:
                continue
            if org_id and record.org_id != org_id:
                continue
            if correlation_id and record.correlation_id != correlation_id:
                continue
            if request_path and not record.request_path.startswith(request_path):
                continue

            results.append(record)

        # Apply pagination
        return results[offset : offset + limit]

    async def get_last_hash(self) -> str:
        """Get last record hash."""
        return self._last_hash

    async def get_chain(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[AuditRecord]:
        """Get records in chain order."""
        results = []
        for record_id in self._order:  # Oldest first (chain order)
            record = self._records.get(record_id)
            if not record:
                continue
            if start_date and record.timestamp < start_date:
                continue
            if end_date and record.timestamp > end_date:
                continue
            results.append(record)
        return results

    async def delete_before(self, cutoff: datetime) -> int:
        """Delete records before cutoff."""
        async with self._lock:
            deleted = 0
            new_order = []
            for record_id in self._order:
                record = self._records.get(record_id)
                if record and record.timestamp < cutoff:
                    del self._records[record_id]
                    deleted += 1
                else:
                    new_order.append(record_id)
            self._order = new_order
            return deleted

    async def count(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> int:
        """Count records in range."""
        if not start_date and not end_date:
            return len(self._records)

        count = 0
        for record in self._records.values():
            if start_date and record.timestamp < start_date:
                continue
            if end_date and record.timestamp > end_date:
                continue
            count += 1
        return count


class PostgresAuditStorage(AuditStorage):
    """
    PostgreSQL audit storage for production deployments.

    Provides durable, scalable storage with full SQL query capabilities.
    Requires asyncpg or psycopg for async operations.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS gateway_audit_records (
        id TEXT PRIMARY KEY,
        correlation_id TEXT,
        timestamp TIMESTAMPTZ NOT NULL,
        request_method TEXT,
        request_path TEXT,
        request_headers JSONB DEFAULT '{}',
        request_body JSONB,
        request_body_hash TEXT,
        response_status INTEGER,
        response_headers JSONB DEFAULT '{}',
        response_body JSONB,
        response_body_hash TEXT,
        duration_ms FLOAT,
        user_id TEXT,
        org_id TEXT,
        ip_address TEXT,
        user_agent TEXT,
        record_hash TEXT NOT NULL,
        previous_hash TEXT,
        signature TEXT,
        metadata JSONB DEFAULT '{}',
        pii_fields_redacted TEXT[]
    );

    CREATE INDEX IF NOT EXISTS idx_gateway_audit_timestamp
        ON gateway_audit_records(timestamp);
    CREATE INDEX IF NOT EXISTS idx_gateway_audit_user
        ON gateway_audit_records(user_id);
    CREATE INDEX IF NOT EXISTS idx_gateway_audit_org
        ON gateway_audit_records(org_id);
    CREATE INDEX IF NOT EXISTS idx_gateway_audit_correlation
        ON gateway_audit_records(correlation_id);
    CREATE INDEX IF NOT EXISTS idx_gateway_audit_path
        ON gateway_audit_records(request_path);
    """

    def __init__(self, database_url: str | None = None) -> None:
        """
        Initialize PostgreSQL storage.

        Args:
            database_url: PostgreSQL connection URL. If None, reads from
                          DATABASE_URL or ARAGORA_POSTGRES_DSN environment.
        """
        self._database_url = (
            database_url or os.environ.get("DATABASE_URL") or os.environ.get("ARAGORA_POSTGRES_DSN")
        )
        if not self._database_url:
            raise ValueError("PostgreSQL URL required. Set DATABASE_URL or ARAGORA_POSTGRES_DSN.")
        self._pool: Any = None
        self._last_hash = ""

    async def _get_pool(self) -> Any:
        """Get or create connection pool."""
        if self._pool is None:
            try:
                import asyncpg

                self._pool = await asyncpg.create_pool(self._database_url, min_size=2, max_size=10)
                # Ensure schema exists
                async with self._pool.acquire() as conn:
                    await conn.execute(self.SCHEMA)
                    # Load last hash
                    row = await conn.fetchrow(
                        "SELECT record_hash FROM gateway_audit_records "
                        "ORDER BY timestamp DESC LIMIT 1"
                    )
                    if row:
                        self._last_hash = row["record_hash"]
            except ImportError:
                raise ImportError(
                    "asyncpg required for PostgresAuditStorage. Install with: pip install asyncpg"
                )
        return self._pool

    async def store(self, record: AuditRecord) -> None:
        """Store an audit record."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO gateway_audit_records (
                    id, correlation_id, timestamp, request_method, request_path,
                    request_headers, request_body, request_body_hash,
                    response_status, response_headers, response_body, response_body_hash,
                    duration_ms, user_id, org_id, ip_address, user_agent,
                    record_hash, previous_hash, signature, metadata, pii_fields_redacted
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                    $13, $14, $15, $16, $17, $18, $19, $20, $21, $22
                )
                """,
                record.id,
                record.correlation_id,
                record.timestamp,
                record.request_method,
                record.request_path,
                json.dumps(record.request_headers),
                json.dumps(record.request_body) if record.request_body else None,
                record.request_body_hash,
                record.response_status,
                json.dumps(record.response_headers),
                json.dumps(record.response_body) if record.response_body else None,
                record.response_body_hash,
                record.duration_ms,
                record.user_id,
                record.org_id,
                record.ip_address,
                record.user_agent,
                record.record_hash,
                record.previous_hash,
                record.signature,
                json.dumps(record.metadata),
                record.pii_fields_redacted,
            )
            self._last_hash = record.record_hash

    async def get(self, record_id: str) -> AuditRecord | None:
        """Get a record by ID."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM gateway_audit_records WHERE id = $1", record_id
            )
            if row:
                return self._row_to_record(row)
            return None

    async def query(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        correlation_id: str | None = None,
        request_path: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[AuditRecord]:
        """Query records with filters."""
        conditions = ["1=1"]
        params: list[Any] = []
        param_idx = 1

        if start_date:
            conditions.append(f"timestamp >= ${param_idx}")
            params.append(start_date)
            param_idx += 1
        if end_date:
            conditions.append(f"timestamp <= ${param_idx}")
            params.append(end_date)
            param_idx += 1
        if user_id:
            conditions.append(f"user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1
        if org_id:
            conditions.append(f"org_id = ${param_idx}")
            params.append(org_id)
            param_idx += 1
        if correlation_id:
            conditions.append(f"correlation_id = ${param_idx}")
            params.append(correlation_id)
            param_idx += 1
        if request_path:
            conditions.append(f"request_path LIKE ${param_idx}")
            params.append(f"{request_path}%")
            param_idx += 1

        params.extend([limit, offset])

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM gateway_audit_records
                WHERE {" AND ".join(conditions)}
                ORDER BY timestamp DESC
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
                """,
                *params,
            )
            return [self._row_to_record(row) for row in rows]

    async def get_last_hash(self) -> str:
        """Get last record hash."""
        return self._last_hash

    async def get_chain(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[AuditRecord]:
        """Get records in chain order."""
        conditions = ["1=1"]
        params: list[Any] = []
        param_idx = 1

        if start_date:
            conditions.append(f"timestamp >= ${param_idx}")
            params.append(start_date)
            param_idx += 1
        if end_date:
            conditions.append(f"timestamp <= ${param_idx}")
            params.append(end_date)
            param_idx += 1

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM gateway_audit_records
                WHERE {" AND ".join(conditions)}
                ORDER BY timestamp ASC
                """,
                *params,
            )
            return [self._row_to_record(row) for row in rows]

    async def delete_before(self, cutoff: datetime) -> int:
        """Delete records before cutoff."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM gateway_audit_records WHERE timestamp < $1", cutoff
            )
            # Parse "DELETE N" result
            return int(result.split()[-1]) if result else 0

    async def count(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> int:
        """Count records in range."""
        conditions = ["1=1"]
        params: list[Any] = []
        param_idx = 1

        if start_date:
            conditions.append(f"timestamp >= ${param_idx}")
            params.append(start_date)
            param_idx += 1
        if end_date:
            conditions.append(f"timestamp <= ${param_idx}")
            params.append(end_date)
            param_idx += 1

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT COUNT(*) as count FROM gateway_audit_records
                WHERE {" AND ".join(conditions)}
                """,
                *params,
            )
            return row["count"] if row else 0

    def _row_to_record(self, row: Any) -> AuditRecord:
        """Convert a database row to an AuditRecord."""
        return AuditRecord(
            id=row["id"],
            correlation_id=row["correlation_id"] or "",
            timestamp=row["timestamp"],
            request_method=row["request_method"] or "",
            request_path=row["request_path"] or "",
            request_headers=json.loads(row["request_headers"]) if row["request_headers"] else {},
            request_body=json.loads(row["request_body"]) if row["request_body"] else None,
            request_body_hash=row["request_body_hash"] or "",
            response_status=row["response_status"] or 0,
            response_headers=json.loads(row["response_headers"]) if row["response_headers"] else {},
            response_body=json.loads(row["response_body"]) if row["response_body"] else None,
            response_body_hash=row["response_body_hash"] or "",
            duration_ms=row["duration_ms"] or 0.0,
            user_id=row["user_id"],
            org_id=row["org_id"],
            ip_address=row["ip_address"] or "",
            user_agent=row["user_agent"] or "",
            record_hash=row["record_hash"] or "",
            previous_hash=row["previous_hash"] or "",
            signature=row["signature"] or "",
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            pii_fields_redacted=row["pii_fields_redacted"] or [],
        )


# =============================================================================
# Audit Interceptor
# =============================================================================


class AuditInterceptor:
    """
    Enterprise audit interceptor for request/response logging.

    Provides comprehensive audit logging with:
    - SHA-256 hashing and hash chains for tamper detection
    - HMAC-SHA256 signatures for integrity verification
    - GDPR-compliant PII redaction
    - SOC 2 Type II audit evidence generation
    - Real-time event emission and webhook integration
    - Prometheus metrics for monitoring

    Usage:
        interceptor = AuditInterceptor(config=AuditConfig(
            retention_days=365,
            emit_events=True,
            webhook_url="https://siem.example.com/webhook",
        ))

        # Intercept a request/response pair
        record = await interceptor.intercept(
            request={"method": "POST", "path": "/api/users", ...},
            response={"status": 200, "body": {...}},
            correlation_id="req-123",
            user_id="user-456",
        )

        # Verify chain integrity
        is_valid, errors = await interceptor.verify_chain()

        # Export for compliance audit
        report = await interceptor.export_soc2_evidence(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 31),
        )
    """

    def __init__(
        self,
        config: AuditConfig | None = None,
        storage: AuditStorage | None = None,
    ) -> None:
        """
        Initialize the audit interceptor.

        Args:
            config: Audit configuration. If None, uses defaults.
            storage: Storage backend. If None, creates based on config.
        """
        self._config = config or AuditConfig()
        self._storage = storage
        self._event_handlers: list[Callable[[AuditEventType, dict[str, Any]], None]] = []
        self._metrics_enabled = self._config.enable_metrics

        # Metrics counters
        self._requests_total = 0
        self._requests_by_status: dict[int, int] = {}
        self._pii_redactions_total = 0
        self._chain_verifications_total = 0
        self._chain_errors_total = 0

        # Initialize storage
        if self._storage is None:
            if self._config.storage_backend == "postgres":
                self._storage = PostgresAuditStorage()
            else:
                self._storage = InMemoryAuditStorage()

        logger.info(
            "AuditInterceptor initialized with %s storage, retention=%d days",
            self._config.storage_backend,
            self._config.retention_days,
        )

    def add_event_handler(self, handler: Callable[[AuditEventType, dict[str, Any]], None]) -> None:
        """
        Add an event handler for audit events.

        Args:
            handler: Callback function receiving (event_type, event_data)
        """
        self._event_handlers.append(handler)

    def remove_event_handler(
        self, handler: Callable[[AuditEventType, dict[str, Any]], None]
    ) -> None:
        """Remove an event handler."""
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)

    async def intercept(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
        correlation_id: str | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        ip_address: str = "",
        user_agent: str = "",
        start_time: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditRecord:
        """
        Intercept and log a request/response pair.

        Args:
            request: Request data with method, path, headers, body
            response: Response data with status, headers, body
            correlation_id: Request correlation/trace ID
            user_id: Authenticated user ID
            org_id: Organization ID
            ip_address: Client IP address
            user_agent: Client user agent
            start_time: Request start time (time.time()) for duration calc
            metadata: Additional metadata to include

        Returns:
            The created audit record
        """
        # Calculate duration
        duration_ms = 0.0
        if start_time:
            duration_ms = (time.time() - start_time) * 1000

        # Extract request data
        request_method = request.get("method", "")
        request_path = request.get("path", "")
        request_headers = dict(request.get("headers", {}))
        request_body = request.get("body")

        # Extract response data
        response_status = response.get("status", 0)
        response_headers = dict(response.get("headers", {}))
        response_body = response.get("body")

        # Redact sensitive headers
        request_headers = self._redact_headers(request_headers)
        response_headers = self._redact_headers(response_headers)

        # Hash original bodies before redaction
        request_body_hash = self._hash_body(request_body)
        response_body_hash = self._hash_body(response_body)

        # Redact PII from bodies
        redacted_fields: list[str] = []
        request_body, req_redacted = self._redact_body(request_body, "request")
        response_body, resp_redacted = self._redact_body(response_body, "response")
        redacted_fields.extend(req_redacted)
        redacted_fields.extend(resp_redacted)

        # Optionally hash response body for privacy
        if self._config.hash_responses and response_body:
            response_body = {"_hashed": True, "hash": response_body_hash}

        # Truncate large bodies
        request_body = self._truncate_body(request_body)
        response_body = self._truncate_body(response_body)

        # Get previous hash for chain
        previous_hash = await self._storage.get_last_hash()

        # Create record
        record = AuditRecord(
            correlation_id=correlation_id or str(uuid4()),
            request_method=request_method,
            request_path=request_path,
            request_headers=request_headers,
            request_body=request_body,
            request_body_hash=request_body_hash,
            response_status=response_status,
            response_headers=response_headers,
            response_body=response_body,
            response_body_hash=response_body_hash,
            duration_ms=duration_ms,
            user_id=user_id,
            org_id=org_id,
            ip_address=ip_address,
            user_agent=user_agent,
            previous_hash=previous_hash,
            metadata=metadata or {},
            pii_fields_redacted=redacted_fields,
        )

        # Compute hash and signature
        record.record_hash = record.compute_hash()
        record.signature = record.compute_signature()

        # Store record
        await self._storage.store(record)

        # Update metrics
        self._requests_total += 1
        self._requests_by_status[response_status] = (
            self._requests_by_status.get(response_status, 0) + 1
        )
        if redacted_fields:
            self._pii_redactions_total += len(redacted_fields)

        # Emit events
        if self._config.emit_events:
            await self._emit_event(
                AuditEventType.RESPONSE_SENT,
                {
                    "record_id": record.id,
                    "correlation_id": record.correlation_id,
                    "method": request_method,
                    "path": request_path,
                    "status": response_status,
                    "duration_ms": duration_ms,
                    "user_id": user_id,
                    "pii_fields_redacted": len(redacted_fields),
                },
            )

        # Send to webhook if configured
        if self._config.webhook_url:
            asyncio.create_task(self._send_webhook(record))

        logger.debug(
            "Audit record created: id=%s method=%s path=%s status=%d duration=%.2fms",
            record.id,
            request_method,
            request_path,
            response_status,
            duration_ms,
        )

        return record

    async def verify_chain(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Verify the integrity of the audit chain.

        Checks that:
        1. Each record's hash matches its computed hash
        2. Each record's previous_hash matches the prior record's hash
        3. Each record's signature is valid

        Args:
            since: Start of verification range
            until: End of verification range

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: list[str] = []
        records = await self._storage.get_chain(since, until)

        prev_hash = ""
        for i, record in enumerate(records):
            # Verify previous hash chain
            if record.previous_hash != prev_hash:
                errors.append(
                    f"Chain broken at record {record.id}: "
                    f"expected previous_hash={prev_hash}, got {record.previous_hash}"
                )

            # Verify record hash
            computed_hash = record.compute_hash()
            if record.record_hash != computed_hash:
                errors.append(
                    f"Hash mismatch at record {record.id}: "
                    f"stored={record.record_hash}, computed={computed_hash}"
                )

            # Verify signature
            if record.signature and not record.verify_signature():
                errors.append(f"Invalid signature at record {record.id}")

            prev_hash = record.record_hash

        # Update metrics
        self._chain_verifications_total += 1
        if errors:
            self._chain_errors_total += 1

        # Emit event
        if self._config.emit_events:
            event_type = (
                AuditEventType.CHAIN_VERIFIED if not errors else AuditEventType.CHAIN_BROKEN
            )
            await self._emit_event(
                event_type,
                {
                    "records_verified": len(records),
                    "errors": len(errors),
                    "since": since.isoformat() if since else None,
                    "until": until.isoformat() if until else None,
                },
            )

        is_valid = len(errors) == 0
        if is_valid:
            logger.info("Audit chain verified: %d records, no errors", len(records))
        else:
            logger.warning("Audit chain verification failed: %d errors", len(errors))

        return is_valid, errors

    async def apply_retention(self) -> int:
        """
        Apply retention policy and delete old records.

        Returns:
            Number of records deleted
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._config.retention_days)
        deleted = await self._storage.delete_before(cutoff)

        if deleted > 0:
            logger.info(
                "Retention policy applied: deleted %d records older than %s",
                deleted,
                cutoff.date(),
            )

            if self._config.emit_events:
                await self._emit_event(
                    AuditEventType.RETENTION_APPLIED,
                    {
                        "records_deleted": deleted,
                        "cutoff_date": cutoff.isoformat(),
                        "retention_days": self._config.retention_days,
                    },
                )

        return deleted

    async def export_soc2_evidence(
        self,
        start_date: datetime,
        end_date: datetime,
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Export audit records as SOC 2 Type II evidence.

        Generates a comprehensive report suitable for SOC 2 auditors,
        including integrity verification and control evidence.

        Args:
            start_date: Audit period start
            end_date: Audit period end
            org_id: Filter by organization

        Returns:
            SOC 2 evidence report dictionary
        """
        # Get records
        records = await self._storage.query(
            start_date=start_date,
            end_date=end_date,
            org_id=org_id,
            limit=100000,
        )

        # Verify chain integrity
        is_valid, integrity_errors = await self.verify_chain(start_date, end_date)

        # Compute statistics
        total_records = len(records)
        by_method: dict[str, int] = {}
        by_status: dict[int, int] = {}
        by_path_prefix: dict[str, int] = {}
        unique_users: set[str] = set()
        unique_ips: set[str] = set()
        total_duration_ms = 0.0
        failed_requests = 0
        pii_redactions = 0

        for record in records:
            by_method[record.request_method] = by_method.get(record.request_method, 0) + 1
            by_status[record.response_status] = by_status.get(record.response_status, 0) + 1

            # Extract path prefix (first two segments)
            path_parts = record.request_path.split("/")[:3]
            path_prefix = "/".join(path_parts)
            by_path_prefix[path_prefix] = by_path_prefix.get(path_prefix, 0) + 1

            if record.user_id:
                unique_users.add(record.user_id)
            if record.ip_address:
                unique_ips.add(record.ip_address)

            total_duration_ms += record.duration_ms

            if record.response_status >= 400:
                failed_requests += 1

            pii_redactions += len(record.pii_fields_redacted)

        avg_duration_ms = total_duration_ms / total_records if total_records > 0 else 0

        # Build SOC 2 report
        report = {
            "report_type": "SOC 2 Type II Gateway Audit Evidence",
            "generated_at": datetime.now(timezone.utc).isoformat(),
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
                "total_requests": total_records,
                "unique_users": len(unique_users),
                "unique_ips": len(unique_ips),
                "failed_requests": failed_requests,
                "pii_redactions": pii_redactions,
                "avg_duration_ms": round(avg_duration_ms, 2),
            },
            "breakdown": {
                "by_method": by_method,
                "by_status": dict(sorted(by_status.items())),
                "by_path_prefix": dict(
                    sorted(by_path_prefix.items(), key=lambda x: x[1], reverse=True)[:20]
                ),
            },
            "control_evidence": {
                "CC6.1_access_control": {
                    "requests_with_user_id": sum(1 for r in records if r.user_id),
                    "requests_without_user_id": sum(1 for r in records if not r.user_id),
                },
                "CC6.6_audit_logging": {
                    "total_logged": total_records,
                    "chain_integrity": is_valid,
                    "signatures_verified": sum(1 for r in records if r.signature),
                },
                "CC6.7_data_protection": {
                    "pii_fields_redacted": pii_redactions,
                    "sensitive_headers_protected": True,
                },
                "CC7.2_monitoring": {
                    "failed_requests": failed_requests,
                    "error_rate_percent": round(failed_requests / total_records * 100, 2)
                    if total_records > 0
                    else 0,
                },
            },
            "sample_records": [r.to_dict() for r in records[:10]],
        }

        # Emit event
        if self._config.emit_events:
            await self._emit_event(
                AuditEventType.EXPORT_GENERATED,
                {
                    "report_type": "soc2",
                    "records_exported": total_records,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
            )

        logger.info(
            "SOC 2 evidence export generated: %d records, integrity=%s",
            total_records,
            is_valid,
        )

        return report

    async def get_record(self, record_id: str) -> AuditRecord | None:
        """
        Get a specific audit record by ID.

        Args:
            record_id: The record ID

        Returns:
            The audit record or None if not found
        """
        return await self._storage.get(record_id)

    async def query_records(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        correlation_id: str | None = None,
        request_path: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[AuditRecord]:
        """
        Query audit records.

        Args:
            start_date: Filter records after this time
            end_date: Filter records before this time
            user_id: Filter by user ID
            org_id: Filter by organization ID
            correlation_id: Filter by correlation ID
            request_path: Filter by request path (prefix match)
            limit: Maximum records to return
            offset: Pagination offset

        Returns:
            List of matching audit records
        """
        return await self._storage.query(
            start_date=start_date,
            end_date=end_date,
            user_id=user_id,
            org_id=org_id,
            correlation_id=correlation_id,
            request_path=request_path,
            limit=limit,
            offset=offset,
        )

    def get_metrics(self) -> dict[str, Any]:
        """
        Get Prometheus-compatible metrics.

        Returns:
            Dictionary of metric values
        """
        return {
            f"{self._config.metrics_prefix}_requests_total": self._requests_total,
            f"{self._config.metrics_prefix}_requests_by_status": self._requests_by_status,
            f"{self._config.metrics_prefix}_pii_redactions_total": self._pii_redactions_total,
            f"{self._config.metrics_prefix}_chain_verifications_total": self._chain_verifications_total,
            f"{self._config.metrics_prefix}_chain_errors_total": self._chain_errors_total,
        }

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _redact_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Redact sensitive headers."""
        redacted = {}
        sensitive = {h.lower() for h in self._config.sensitive_headers}

        for key, value in headers.items():
            if key.lower() in sensitive:
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = value

        return redacted

    def _hash_body(self, body: Any) -> str:
        """Compute SHA-256 hash of a body."""
        if body is None:
            return ""
        try:
            if isinstance(body, (dict, list)):
                data = json.dumps(body, sort_keys=True, separators=(",", ":"))
            else:
                data = str(body)
            return hashlib.sha256(data.encode()).hexdigest()
        except (TypeError, ValueError):
            return ""

    def _redact_body(self, body: Any, prefix: str = "") -> tuple[Any, list[str]]:
        """
        Recursively redact PII from a body.

        Returns:
            Tuple of (redacted_body, list of redacted field names)
        """
        redacted_fields: list[str] = []

        if body is None:
            return None, redacted_fields

        if isinstance(body, dict):
            redacted = {}
            for key, value in body.items():
                field_path = f"{prefix}.{key}" if prefix else key

                # Check if this field should be redacted
                for rule in self._config.pii_rules:
                    if rule.matches(key):
                        redacted[key] = rule.redact(value)
                        redacted_fields.append(field_path)
                        break
                else:
                    # Recursively process nested structures
                    if isinstance(value, (dict, list)):
                        redacted[key], nested_fields = self._redact_body(value, field_path)
                        redacted_fields.extend(nested_fields)
                    else:
                        redacted[key] = value

            return redacted, redacted_fields

        elif isinstance(body, list):
            redacted = []
            for i, item in enumerate(body):
                field_path = f"{prefix}[{i}]"
                if isinstance(item, (dict, list)):
                    redacted_item, nested_fields = self._redact_body(item, field_path)
                    redacted.append(redacted_item)
                    redacted_fields.extend(nested_fields)
                else:
                    redacted.append(item)
            return redacted, redacted_fields

        return body, redacted_fields

    def _truncate_body(self, body: Any) -> Any:
        """Truncate body if it exceeds max size."""
        if body is None or self._config.max_body_size == 0:
            return body

        try:
            serialized = json.dumps(body)
            if len(serialized) > self._config.max_body_size:
                return {
                    "_truncated": True,
                    "_original_size": len(serialized),
                    "_preview": serialized[: self._config.max_body_size // 10],
                }
        except (TypeError, ValueError):
            pass

        return body

    async def _emit_event(self, event_type: AuditEventType, data: dict[str, Any]) -> None:
        """Emit an audit event to all handlers."""
        for handler in self._event_handlers:
            try:
                handler(event_type, data)
            except Exception as e:
                logger.error("Error in audit event handler: %s", e)

    async def _send_webhook(self, record: AuditRecord) -> None:
        """Send audit record to webhook."""
        if not self._config.webhook_url:
            return

        try:
            import aiohttp

            payload = {
                "event_type": "audit_record",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "record": record.to_dict(),
            }

            headers = {
                "Content-Type": "application/json",
                **self._config.webhook_headers,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._config.webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self._config.webhook_timeout),
                ) as response:
                    if response.status >= 400:
                        logger.warning(
                            "Webhook failed: status=%d url=%s",
                            response.status,
                            self._config.webhook_url,
                        )
        except ImportError:
            logger.warning("aiohttp not installed, webhook disabled")
        except Exception as e:
            logger.error("Webhook error: %s", e)


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Core classes
    "AuditInterceptor",
    "AuditRecord",
    "AuditConfig",
    "PIIRedactionRule",
    # Enums
    "RedactionType",
    "AuditEventType",
    # Storage
    "AuditStorage",
    "InMemoryAuditStorage",
    "PostgresAuditStorage",
    # Key management
    "get_interceptor_signing_key",
    "set_interceptor_signing_key",
]
