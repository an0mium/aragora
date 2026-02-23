"""
Integration store data models and encryption helpers.

Contains:
- UserIdMapping and IntegrationConfig dataclasses
- Encryption/decryption helpers for sensitive settings
- Storage key helper
- Metrics recording helpers
- Type definitions and constants
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Import encryption (optional - graceful degradation if not available)
get_encryption_service: Any
is_encryption_required: Any
EncryptionError: Any

try:
    from aragora.security.encryption import (
        get_encryption_service,
        is_encryption_required,
        EncryptionError,
        CRYPTO_AVAILABLE,
    )
except ImportError:
    CRYPTO_AVAILABLE = False

    def _fallback_get_encryption_service() -> Any:
        raise RuntimeError("Encryption not available")

    def _fallback_is_encryption_required() -> bool:
        """Fallback when security module unavailable - still check env vars."""
        import os

        if os.environ.get("ARAGORA_ENCRYPTION_REQUIRED", "").lower() in ("true", "1", "yes"):
            return True
        if os.environ.get("ARAGORA_ENV") == "production":
            return True
        return False

    class _FallbackEncryptionError(Exception):
        """Fallback exception when security module unavailable."""

        def __init__(self, operation: str, reason: str, store: str = ""):
            self.operation = operation
            self.reason = reason
            self.store = store
            super().__init__(
                f"Encryption {operation} failed in {store}: {reason}. "
                f"Set ARAGORA_ENCRYPTION_REQUIRED=false to allow plaintext fallback."
            )

    get_encryption_service = _fallback_get_encryption_service
    is_encryption_required = _fallback_is_encryption_required
    EncryptionError = _FallbackEncryptionError


# =============================================================================
# Metrics Recording Helpers
# =============================================================================


def _record_user_mapping_operation(operation: str, platform: str, found: bool) -> None:
    """Record user mapping operation metric if available."""
    try:
        import aragora.observability.metrics as metrics_module

        fn = getattr(metrics_module, "record_user_mapping_operation", None)
        if fn is not None:
            fn(operation, platform, found)
    except ImportError:
        pass


def _record_user_mapping_cache_hit(platform: str) -> None:
    """Record user mapping cache hit metric if available."""
    try:
        import aragora.observability.metrics as metrics_module

        fn = getattr(metrics_module, "record_user_mapping_cache_hit", None)
        if fn is not None:
            fn(platform)
    except ImportError:
        pass


def _record_user_mapping_cache_miss(platform: str) -> None:
    """Record user mapping cache miss metric if available."""
    try:
        import aragora.observability.metrics as metrics_module

        fn = getattr(metrics_module, "record_user_mapping_cache_miss", None)
        if fn is not None:
            fn(platform)
    except ImportError:
        pass


# =============================================================================
# Integration Types and Constants
# =============================================================================

IntegrationType = Literal["slack", "discord", "telegram", "email", "teams", "whatsapp", "matrix"]

VALID_INTEGRATION_TYPES: set[str] = {
    "slack",
    "discord",
    "telegram",
    "email",
    "teams",
    "whatsapp",
    "matrix",
}

# Sensitive keys that should be encrypted or masked
SENSITIVE_KEYS = frozenset(
    [
        "access_token",
        "refresh_token",
        "api_key",
        "bot_token",
        "webhook_url",
        "secret",
        "password",
        "auth_token",
        "sendgrid_api_key",
        "ses_secret_access_key",
        "twilio_auth_token",
        "smtp_password",
    ]
)


# =============================================================================
# Encryption Helpers
# =============================================================================


def _encrypt_settings(
    settings: dict[str, Any],
    user_id: str = "default",
    integration_type: str = "",
) -> dict[str, Any]:
    """
    Encrypt sensitive keys in settings dict before storage.

    Uses Associated Authenticated Data (AAD) to bind ciphertext to user/integration
    context, preventing cross-user/integration attacks.

    Raises:
        EncryptionError: If encryption fails and ARAGORA_ENCRYPTION_REQUIRED is True.
    """
    if not settings:
        return settings

    # Find keys that need encryption and have values
    keys_to_encrypt = [k for k in SENSITIVE_KEYS if k in settings and settings[k]]
    if not keys_to_encrypt:
        return settings

    if not CRYPTO_AVAILABLE:
        if is_encryption_required():
            raise EncryptionError(
                "encrypt",
                "cryptography library not available",
                "integration_store",
            )
        return settings

    try:
        service = get_encryption_service()
        # AAD binds ciphertext to this specific user + integration
        aad = f"{user_id}:{integration_type}"
        encrypted = service.encrypt_fields(settings, keys_to_encrypt, aad)
        logger.debug("Encrypted %s sensitive fields for %s", len(keys_to_encrypt), integration_type)
        return encrypted
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        if is_encryption_required():
            raise EncryptionError(
                "encrypt",
                str(e),
                "integration_store",
            ) from e
        logger.warning("Encryption unavailable, storing unencrypted: %s", e)
        return settings
    except RuntimeError as e:
        if is_encryption_required():
            raise EncryptionError(
                "encrypt",
                str(e),
                "integration_store",
            ) from e
        logger.warning("Encryption service error, storing unencrypted: %s", e)
        return settings


def _decrypt_settings(
    settings: dict[str, Any],
    user_id: str = "default",
    integration_type: str = "",
) -> dict[str, Any]:
    """
    Decrypt sensitive keys, handling legacy unencrypted data.

    AAD must match what was used during encryption.
    """
    if not CRYPTO_AVAILABLE or not settings:
        return settings

    # Check for encryption markers - if none present, it's legacy data
    encrypted_keys = [
        k
        for k in SENSITIVE_KEYS
        if k in settings and isinstance(settings.get(k), dict) and settings[k].get("_encrypted")
    ]
    if not encrypted_keys:
        return settings  # Legacy unencrypted data - return as-is

    try:
        service = get_encryption_service()
        aad = f"{user_id}:{integration_type}"
        decrypted = service.decrypt_fields(settings, encrypted_keys, aad)
        logger.debug("Decrypted %s fields for %s", len(encrypted_keys), integration_type)
        return decrypted
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
        logger.warning("Decryption failed for %s: %s", integration_type, e)
        return settings
    except Exception as e:  # noqa: BLE001 - cryptography errors don't subclass standard types
        # Catch cryptography-specific errors (e.g. InvalidTag from AAD mismatch)
        if type(e).__name__ in ("InvalidTag", "InvalidSignature"):
            logger.warning("Decryption AAD mismatch for %s: %s", integration_type, e)
            return settings
        raise


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class UserIdMapping:
    """Cross-platform user identity mapping."""

    email: str
    platform: str  # "slack", "discord", "teams", etc.
    platform_user_id: str
    display_name: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    user_id: str = "default"  # Owner/tenant

    def to_dict(self) -> dict:
        """Convert to dict."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> UserIdMapping:
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def from_row(cls, row: tuple) -> UserIdMapping:
        """Create from database row."""
        return cls(
            email=row[0],
            platform=row[1],
            platform_user_id=row[2],
            display_name=row[3],
            created_at=row[4],
            updated_at=row[5],
            user_id=row[6],
        )


@dataclass
class IntegrationConfig:
    """Configuration for a chat platform integration."""

    type: str
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Notification settings
    notify_on_consensus: bool = True
    notify_on_debate_end: bool = True
    notify_on_error: bool = False
    notify_on_leaderboard: bool = False

    # Provider-specific settings (stored as dict)
    settings: dict[str, Any] = field(default_factory=dict)

    # Delivery tracking
    messages_sent: int = 0
    errors_24h: int = 0
    last_activity: float | None = None
    last_error: str | None = None

    # Owner (for multi-tenant)
    user_id: str | None = None
    workspace_id: str | None = None

    def to_dict(self, include_secrets: bool = False) -> dict:
        """Convert to dict, optionally excluding secrets."""
        result = asdict(self)
        if not include_secrets:
            settings = result.get("settings", {})
            for key in SENSITIVE_KEYS:
                if key in settings and settings[key]:
                    settings[key] = "••••••••"
            result["settings"] = settings
        return result

    def to_json(self) -> str:
        """Serialize to JSON for storage."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> IntegrationConfig:
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def from_row(cls, row: tuple) -> IntegrationConfig:
        """Create from database row (settings decryption done at store level)."""
        return cls(
            type=row[0],
            enabled=bool(row[1]),
            created_at=row[2],
            updated_at=row[3],
            notify_on_consensus=bool(row[4]),
            notify_on_debate_end=bool(row[5]),
            notify_on_error=bool(row[6]),
            notify_on_leaderboard=bool(row[7]),
            settings=json.loads(row[8]) if row[8] else {},
            messages_sent=row[9] or 0,
            errors_24h=row[10] or 0,
            last_activity=row[11],
            last_error=row[12],
            user_id=row[13],
            workspace_id=row[14],
        )

    @property
    def status(self) -> str:
        """Get integration status."""
        if not self.enabled:
            return "disconnected"
        if self.errors_24h > 5:
            return "degraded"
        if self.last_activity:
            return "connected"
        return "not_configured"


def _make_key(integration_type: str, user_id: str = "default") -> str:
    """Generate storage key for integration."""
    return f"{user_id}:{integration_type}"


__all__ = [
    "IntegrationConfig",
    "IntegrationType",
    "VALID_INTEGRATION_TYPES",
    "SENSITIVE_KEYS",
    "UserIdMapping",
    "CRYPTO_AVAILABLE",
    "_encrypt_settings",
    "_decrypt_settings",
    "_make_key",
    "_record_user_mapping_operation",
    "_record_user_mapping_cache_hit",
    "_record_user_mapping_cache_miss",
]
