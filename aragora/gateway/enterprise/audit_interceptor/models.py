"""
Data models for the Audit Interceptor.

Contains:
- PIIRedactionRule: Rule for redacting PII fields
- AuditConfig: Configuration for the interceptor
- AuditRecord: Audit record capturing request/response pairs
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from collections.abc import Callable
from uuid import uuid4

from .enums import RedactionType, get_interceptor_signing_key


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
    def from_dict(cls, data: dict[str, Any]) -> AuditRecord:
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


__all__ = [
    "PIIRedactionRule",
    "AuditConfig",
    "AuditRecord",
]
