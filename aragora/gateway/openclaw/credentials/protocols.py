"""
OpenClaw Credential Vault - Protocol definitions.
"""

from __future__ import annotations

from typing import Any, Protocol


class KMSProviderProtocol(Protocol):
    """Protocol for KMS providers."""

    async def get_encryption_key(self, key_id: str) -> bytes:
        """Get or generate an encryption key."""
        ...

    async def decrypt_data_key(self, encrypted_key: bytes, key_id: str) -> bytes:
        """Decrypt a data key using the master key."""
        ...

    async def encrypt_data_key(self, plaintext_key: bytes, key_id: str) -> bytes:
        """Encrypt a data key using the master key."""
        ...


class AuditLoggerProtocol(Protocol):
    """Protocol for audit logging."""

    async def log_event(
        self,
        event_type: str,
        actor_id: str,
        tenant_id: str | None = None,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
        severity: str = "info",
    ) -> None:
        """Log an audit event."""
        ...


class AuthorizationContextProtocol(Protocol):
    """Protocol for authorization context."""

    user_id: str
    org_id: str | None
    roles: set[str]
    permissions: set[str]

    def has_permission(self, permission_key: str) -> bool:
        """Check if context has a permission."""
        ...

    def has_role(self, role_name: str) -> bool:
        """Check if context has a specific role."""
        ...
