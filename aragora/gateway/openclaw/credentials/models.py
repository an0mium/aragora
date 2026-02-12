"""
OpenClaw Credential Vault - Data model classes.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .enums import CredentialType


@dataclass
class CredentialMetadata:
    """
    Metadata tracking for a stored credential.

    Attributes:
        created_at: When the credential was first stored
        created_by: User ID who created the credential
        rotated_at: When the credential was last rotated (None if never)
        rotated_by: User ID who performed the last rotation
        expires_at: When the credential expires (None if never)
        access_count: Number of times the credential has been retrieved
        last_accessed_at: When the credential was last retrieved
        last_accessed_by: User ID who last accessed the credential
        version: Version number (incremented on rotation)
        tags: Custom tags for categorization
        description: Human-readable description
    """

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str | None = None
    rotated_at: datetime | None = None
    rotated_by: str | None = None
    expires_at: datetime | None = None
    access_count: int = 0
    last_accessed_at: datetime | None = None
    last_accessed_by: str | None = None
    version: int = 1
    tags: list[str] = field(default_factory=list)
    description: str = ""

    @property
    def is_expired(self) -> bool:
        """Check if the credential has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def days_until_expiry(self) -> int | None:
        """Get days until credential expires, or None if no expiry."""
        if self.expires_at is None:
            return None
        delta = self.expires_at - datetime.now(timezone.utc)
        return max(0, delta.days)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "rotated_at": self.rotated_at.isoformat() if self.rotated_at else None,
            "rotated_by": self.rotated_by,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "access_count": self.access_count,
            "last_accessed_at": (
                self.last_accessed_at.isoformat() if self.last_accessed_at else None
            ),
            "last_accessed_by": self.last_accessed_by,
            "version": self.version,
            "tags": self.tags,
            "description": self.description,
            "is_expired": self.is_expired,
            "days_until_expiry": self.days_until_expiry,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CredentialMetadata:
        """Create from dictionary."""

        def parse_dt(val: str | None) -> datetime | None:
            if val is None:
                return None
            return datetime.fromisoformat(val.replace("Z", "+00:00"))

        return cls(
            created_at=parse_dt(data.get("created_at")) or datetime.now(timezone.utc),
            created_by=data.get("created_by"),
            rotated_at=parse_dt(data.get("rotated_at")),
            rotated_by=data.get("rotated_by"),
            expires_at=parse_dt(data.get("expires_at")),
            access_count=data.get("access_count", 0),
            last_accessed_at=parse_dt(data.get("last_accessed_at")),
            last_accessed_by=data.get("last_accessed_by"),
            version=data.get("version", 1),
            tags=data.get("tags", []),
            description=data.get("description", ""),
        )


@dataclass
class RotationPolicy:
    """
    Policy for automatic credential rotation.

    Attributes:
        interval_days: Days between rotations (0 = manual only)
        notify_before_days: Days before expiry to send alert
        auto_rotate: Whether to rotate automatically when due
        on_access_count: Rotate after N accesses (0 = disabled)
        on_compromise: Rotate immediately if compromise suspected
        require_approval: Require manual approval for rotation
        notify_channels: Channels to notify (email, slack, webhook)
    """

    interval_days: int = 90
    notify_before_days: int = 14
    auto_rotate: bool = False
    on_access_count: int = 0
    on_compromise: bool = True
    require_approval: bool = False
    notify_channels: list[str] = field(default_factory=list)

    def is_rotation_due(self, metadata: CredentialMetadata) -> bool:
        """Check if credential rotation is due based on policy."""
        if self.interval_days <= 0:
            return False

        # Check time-based rotation
        last_rotation = metadata.rotated_at or metadata.created_at
        days_since = (datetime.now(timezone.utc) - last_rotation).days
        if days_since >= self.interval_days:
            return True

        # Check access-count based rotation
        if self.on_access_count > 0 and metadata.access_count >= self.on_access_count:
            return True

        return False

    def needs_expiry_alert(self, metadata: CredentialMetadata) -> bool:
        """Check if expiry alert should be sent."""
        if self.notify_before_days <= 0:
            return False
        days_left = metadata.days_until_expiry
        if days_left is None:
            return False
        return days_left <= self.notify_before_days

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "interval_days": self.interval_days,
            "notify_before_days": self.notify_before_days,
            "auto_rotate": self.auto_rotate,
            "on_access_count": self.on_access_count,
            "on_compromise": self.on_compromise,
            "require_approval": self.require_approval,
            "notify_channels": self.notify_channels,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RotationPolicy:
        """Create from dictionary."""
        return cls(
            interval_days=data.get("interval_days", 90),
            notify_before_days=data.get("notify_before_days", 14),
            auto_rotate=data.get("auto_rotate", False),
            on_access_count=data.get("on_access_count", 0),
            on_compromise=data.get("on_compromise", True),
            require_approval=data.get("require_approval", False),
            notify_channels=data.get("notify_channels", []),
        )

    @classmethod
    def strict(cls) -> RotationPolicy:
        """Create a strict rotation policy for high-security credentials."""
        return cls(
            interval_days=30,
            notify_before_days=7,
            auto_rotate=False,
            on_access_count=1000,
            on_compromise=True,
            require_approval=True,
        )

    @classmethod
    def standard(cls) -> RotationPolicy:
        """Create a standard rotation policy."""
        return cls(
            interval_days=90,
            notify_before_days=14,
            auto_rotate=True,
            on_access_count=0,
            on_compromise=True,
            require_approval=False,
        )

    @classmethod
    def relaxed(cls) -> RotationPolicy:
        """Create a relaxed rotation policy for development."""
        return cls(
            interval_days=365,
            notify_before_days=30,
            auto_rotate=False,
            on_access_count=0,
            on_compromise=True,
            require_approval=False,
        )


@dataclass
class StoredCredential:
    """
    A credential stored in the vault.

    Attributes:
        credential_id: Unique identifier for the credential
        tenant_id: Tenant that owns this credential
        framework: External framework the credential is for
        credential_type: Type of credential (api_key, oauth_token, etc.)
        encrypted_value: AES-256-GCM encrypted credential value
        encryption_key_id: ID of the key used for encryption
        metadata: Credential metadata (created_at, access_count, etc.)
        rotation_policy: Automatic rotation policy
        allowed_agents: Agent names allowed to use this credential (empty = all)
        allowed_capabilities: Capabilities that can use this credential
    """

    credential_id: str
    tenant_id: str
    framework: str
    credential_type: CredentialType
    encrypted_value: bytes
    encryption_key_id: str
    metadata: CredentialMetadata = field(default_factory=CredentialMetadata)
    rotation_policy: RotationPolicy = field(default_factory=RotationPolicy)
    allowed_agents: list[str] = field(default_factory=list)
    allowed_capabilities: list[str] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        """Check if credential has expired."""
        return self.metadata.is_expired

    @property
    def needs_rotation(self) -> bool:
        """Check if credential needs rotation based on policy."""
        return self.rotation_policy.is_rotation_due(self.metadata)

    def to_dict(self, include_encrypted: bool = False) -> dict[str, Any]:
        """Convert to dictionary for serialization (excludes encrypted value by default)."""
        result = {
            "credential_id": self.credential_id,
            "tenant_id": self.tenant_id,
            "framework": self.framework,
            "credential_type": self.credential_type.value,
            "encryption_key_id": self.encryption_key_id,
            "metadata": self.metadata.to_dict(),
            "rotation_policy": self.rotation_policy.to_dict(),
            "allowed_agents": self.allowed_agents,
            "allowed_capabilities": self.allowed_capabilities,
            "is_expired": self.is_expired,
            "needs_rotation": self.needs_rotation,
        }
        if include_encrypted:
            result["encrypted_value"] = base64.b64encode(self.encrypted_value).decode()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StoredCredential:
        """Create from dictionary."""
        encrypted = data.get("encrypted_value", b"")
        if isinstance(encrypted, str):
            encrypted = base64.b64decode(encrypted)

        return cls(
            credential_id=data["credential_id"],
            tenant_id=data["tenant_id"],
            framework=data["framework"],
            credential_type=CredentialType(data["credential_type"]),
            encrypted_value=encrypted,
            encryption_key_id=data.get("encryption_key_id", "default"),
            metadata=CredentialMetadata.from_dict(data.get("metadata", {})),
            rotation_policy=RotationPolicy.from_dict(data.get("rotation_policy", {})),
            allowed_agents=data.get("allowed_agents", []),
            allowed_capabilities=data.get("allowed_capabilities", []),
        )
