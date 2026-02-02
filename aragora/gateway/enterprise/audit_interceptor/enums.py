"""
Enums and constants for the Audit Interceptor.

Contains:
- RedactionType: PII redaction strategies
- AuditEventType: Types of audit events
- Signing key management functions
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
from enum import Enum

logger = logging.getLogger(__name__)


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


__all__ = [
    "RedactionType",
    "AuditEventType",
    "get_interceptor_signing_key",
    "set_interceptor_signing_key",
]
