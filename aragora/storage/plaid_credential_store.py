"""
Plaid Credential Storage with Encryption.

Provides persistent storage for Plaid access tokens and item credentials
with AES-256-GCM encryption using the centralized encryption service.

Security Features:
- AES-256-GCM authenticated encryption for access tokens
- User/tenant ID bound as Associated Authenticated Data (AAD)
- Automatic encryption in production mode
- Graceful fallback with warnings in development

Usage:
    from aragora.storage.plaid_credential_store import get_plaid_credential_store

    store = get_plaid_credential_store()
    await store.save_credentials(credentials)
    credentials = await store.get_credentials(user_id, tenant_id, item_id)
"""

from __future__ import annotations

import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aragora.config import resolve_db_path

logger = logging.getLogger(__name__)

# Import encryption (optional - graceful degradation if not available)
get_encryption_service: Any = None
is_encryption_required: Any = None
EncryptionError: Any = Exception
CRYPTO_AVAILABLE = False

try:
    from aragora.security.encryption import (
        CRYPTO_AVAILABLE,
        EncryptionError,
        get_encryption_service,
        is_encryption_required,
    )
except ImportError:
    CRYPTO_AVAILABLE = False

    def _fb_get_encryption_service() -> Any:
        raise RuntimeError("Encryption not available")

    def _fb_is_encryption_required() -> bool:
        """Fallback when security module unavailable - still check env vars."""
        if os.environ.get("ARAGORA_ENCRYPTION_REQUIRED", "").lower() in ("true", "1", "yes"):
            return True
        if os.environ.get("ARAGORA_ENV") == "production":
            return True
        return False

    class _FBEncryptionError(Exception):
        """Fallback exception when security module unavailable."""

        def __init__(self, operation: str, reason: str, store: str = ""):
            self.operation = operation
            self.reason = reason
            self.store = store
            super().__init__(
                f"Encryption {operation} failed in {store}: {reason}. "
                f"Set ARAGORA_ENCRYPTION_REQUIRED=false to allow plaintext fallback."
            )

    get_encryption_service = _fb_get_encryption_service
    is_encryption_required = _fb_is_encryption_required
    EncryptionError = _FBEncryptionError


def _encrypt_token(token: str, aad: str = "") -> str:
    """
    Encrypt a Plaid access token for storage.

    Uses AAD (Associated Authenticated Data) to bind the ciphertext
    to a specific user/tenant, preventing cross-user token attacks.

    Raises:
        EncryptionError: If encryption fails and ARAGORA_ENCRYPTION_REQUIRED is True.
    """
    if not token:
        return token

    if not CRYPTO_AVAILABLE:
        if is_encryption_required():
            raise EncryptionError(
                "encrypt",
                "cryptography library not available",
                "plaid_credential_store",
            )
        logger.warning(
            "SECURITY WARNING: Storing Plaid token unencrypted - cryptography not available"
        )
        return token

    try:
        service = get_encryption_service()
        encrypted = service.encrypt(token, associated_data=aad if aad else None)
        return encrypted.to_base64()
    except (ValueError, RuntimeError, OSError) as e:
        if is_encryption_required():
            raise EncryptionError(
                "encrypt",
                str(e),
                "plaid_credential_store",
            ) from e
        logger.warning(f"Plaid token encryption failed, storing unencrypted: {e}")
        return token


def _decrypt_token(encrypted_token: str, aad: str = "") -> str:
    """
    Decrypt a Plaid access token.

    AAD must match what was used during encryption.
    Handles legacy unencrypted tokens gracefully.
    """
    if not CRYPTO_AVAILABLE or not encrypted_token:
        return encrypted_token

    # Check if it looks like an encrypted value
    # EncryptedData always starts with version byte 0x01 which encodes to "A" in base64
    if not encrypted_token.startswith("A"):
        return encrypted_token  # Legacy unencrypted token

    try:
        service = get_encryption_service()
        return service.decrypt_string(encrypted_token, associated_data=aad if aad else None)
    except (ValueError, RuntimeError, OSError) as e:
        # Could be a legacy plain token that happens to start with "A"
        logger.warning(f"Plaid token decryption failed (may be legacy token): {e}")
        return encrypted_token


class PlaidCredentialStore(ABC):
    """Abstract base class for Plaid credential storage."""

    @abstractmethod
    async def save_credentials(
        self,
        user_id: str,
        tenant_id: str,
        item_id: str,
        access_token: str,
        institution_id: str,
        institution_name: str,
    ) -> None:
        """Save Plaid credentials with encrypted access token."""
        pass

    @abstractmethod
    async def get_credentials(
        self,
        user_id: str,
        tenant_id: str,
        item_id: str,
    ) -> dict[str, Any] | None:
        """Retrieve Plaid credentials with decrypted access token."""
        pass

    @abstractmethod
    async def delete_credentials(
        self,
        user_id: str,
        tenant_id: str,
        item_id: str,
    ) -> bool:
        """Delete Plaid credentials."""
        pass

    @abstractmethod
    async def list_credentials(
        self,
        user_id: str,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        """List all Plaid credentials for a user."""
        pass

    @abstractmethod
    async def update_last_sync(
        self,
        user_id: str,
        tenant_id: str,
        item_id: str,
    ) -> None:
        """Update the last sync timestamp."""
        pass


class SQLitePlaidCredentialStore(PlaidCredentialStore):
    """SQLite-backed Plaid credential store with encryption."""

    def __init__(self, db_path: str | None = None):
        """Initialize the store.

        Args:
            db_path: Path to SQLite database file. Defaults to ARAGORA_DATA_DIR/plaid_credentials.db
        """
        if db_path is None:
            db_path = "plaid_credentials.db"

        self._db_path = resolve_db_path(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS plaid_credentials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                access_token_encrypted TEXT NOT NULL,
                institution_id TEXT NOT NULL,
                institution_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_sync TEXT,
                UNIQUE(user_id, tenant_id, item_id)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_plaid_user_tenant
            ON plaid_credentials(user_id, tenant_id)
            """
        )
        conn.commit()

    def _make_aad(self, user_id: str, tenant_id: str, item_id: str) -> str:
        """Create AAD binding token to specific user/tenant/item."""
        return f"plaid:{tenant_id}:{user_id}:{item_id}"

    async def save_credentials(
        self,
        user_id: str,
        tenant_id: str,
        item_id: str,
        access_token: str,
        institution_id: str,
        institution_name: str,
    ) -> None:
        """Save Plaid credentials with encrypted access token."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        aad = self._make_aad(user_id, tenant_id, item_id)

        # Encrypt the access token
        encrypted_token = _encrypt_token(access_token, aad)

        conn.execute(
            """
            INSERT OR REPLACE INTO plaid_credentials (
                user_id, tenant_id, item_id,
                access_token_encrypted,
                institution_id, institution_name,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                tenant_id,
                item_id,
                encrypted_token,
                institution_id,
                institution_name,
                now,
                now,
            ),
        )
        conn.commit()
        logger.info(
            f"[PlaidStore] Saved credentials for {institution_name} (item: {item_id[:8]}...)"
        )

    async def get_credentials(
        self,
        user_id: str,
        tenant_id: str,
        item_id: str,
    ) -> dict[str, Any] | None:
        """Retrieve Plaid credentials with decrypted access token."""
        conn = self._get_conn()
        aad = self._make_aad(user_id, tenant_id, item_id)

        row = conn.execute(
            """
            SELECT * FROM plaid_credentials
            WHERE user_id = ? AND tenant_id = ? AND item_id = ?
            """,
            (user_id, tenant_id, item_id),
        ).fetchone()

        if not row:
            return None

        return {
            "user_id": row["user_id"],
            "tenant_id": row["tenant_id"],
            "item_id": row["item_id"],
            "access_token": _decrypt_token(row["access_token_encrypted"], aad),
            "institution_id": row["institution_id"],
            "institution_name": row["institution_name"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_sync": row["last_sync"],
        }

    async def delete_credentials(
        self,
        user_id: str,
        tenant_id: str,
        item_id: str,
    ) -> bool:
        """Delete Plaid credentials."""
        conn = self._get_conn()

        cursor = conn.execute(
            """
            DELETE FROM plaid_credentials
            WHERE user_id = ? AND tenant_id = ? AND item_id = ?
            """,
            (user_id, tenant_id, item_id),
        )
        conn.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"[PlaidStore] Deleted credentials for item: {item_id[:8]}...")
        return deleted

    async def list_credentials(
        self,
        user_id: str,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        """List all Plaid credentials for a user (without access tokens)."""
        conn = self._get_conn()

        rows = conn.execute(
            """
            SELECT user_id, tenant_id, item_id, institution_id, institution_name,
                   created_at, updated_at, last_sync
            FROM plaid_credentials
            WHERE user_id = ? AND tenant_id = ?
            ORDER BY created_at DESC
            """,
            (user_id, tenant_id),
        ).fetchall()

        return [
            {
                "user_id": row["user_id"],
                "tenant_id": row["tenant_id"],
                "item_id": row["item_id"],
                "institution_id": row["institution_id"],
                "institution_name": row["institution_name"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "last_sync": row["last_sync"],
            }
            for row in rows
        ]

    async def update_last_sync(
        self,
        user_id: str,
        tenant_id: str,
        item_id: str,
    ) -> None:
        """Update the last sync timestamp."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()

        conn.execute(
            """
            UPDATE plaid_credentials
            SET last_sync = ?, updated_at = ?
            WHERE user_id = ? AND tenant_id = ? AND item_id = ?
            """,
            (now, now, user_id, tenant_id, item_id),
        )
        conn.commit()


class InMemoryPlaidCredentialStore(PlaidCredentialStore):
    """In-memory Plaid credential store (for testing only)."""

    def __init__(self) -> None:
        self._credentials: dict[str, dict[str, Any]] = {}

    def _make_key(self, user_id: str, tenant_id: str, item_id: str) -> str:
        return f"{tenant_id}:{user_id}:{item_id}"

    async def save_credentials(
        self,
        user_id: str,
        tenant_id: str,
        item_id: str,
        access_token: str,
        institution_id: str,
        institution_name: str,
    ) -> None:
        key = self._make_key(user_id, tenant_id, item_id)
        now = datetime.now(timezone.utc).isoformat()
        aad = f"plaid:{tenant_id}:{user_id}:{item_id}"

        self._credentials[key] = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "item_id": item_id,
            "access_token_encrypted": _encrypt_token(access_token, aad),
            "institution_id": institution_id,
            "institution_name": institution_name,
            "created_at": now,
            "updated_at": now,
            "last_sync": None,
        }

    async def get_credentials(
        self,
        user_id: str,
        tenant_id: str,
        item_id: str,
    ) -> dict[str, Any] | None:
        key = self._make_key(user_id, tenant_id, item_id)
        cred = self._credentials.get(key)

        if not cred:
            return None

        aad = f"plaid:{tenant_id}:{user_id}:{item_id}"
        return {
            **cred,
            "access_token": _decrypt_token(cred["access_token_encrypted"], aad),
        }

    async def delete_credentials(
        self,
        user_id: str,
        tenant_id: str,
        item_id: str,
    ) -> bool:
        key = self._make_key(user_id, tenant_id, item_id)
        if key in self._credentials:
            del self._credentials[key]
            return True
        return False

    async def list_credentials(
        self,
        user_id: str,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        prefix = f"{tenant_id}:{user_id}:"
        return [
            {k: v for k, v in cred.items() if k != "access_token_encrypted"}
            for key, cred in self._credentials.items()
            if key.startswith(prefix)
        ]

    async def update_last_sync(
        self,
        user_id: str,
        tenant_id: str,
        item_id: str,
    ) -> None:
        key = self._make_key(user_id, tenant_id, item_id)
        if key in self._credentials:
            now = datetime.now(timezone.utc).isoformat()
            self._credentials[key]["last_sync"] = now
            self._credentials[key]["updated_at"] = now


# =============================================================================
# Singleton Management
# =============================================================================

_plaid_credential_store: PlaidCredentialStore | None = None


def get_plaid_credential_store() -> PlaidCredentialStore:
    """Get the global Plaid credential store instance."""
    global _plaid_credential_store

    if _plaid_credential_store is None:
        # Use in-memory for testing, SQLite otherwise
        if os.environ.get("ARAGORA_TEST_MODE", "").lower() in ("1", "true", "yes"):
            _plaid_credential_store = InMemoryPlaidCredentialStore()
            logger.info("[PlaidStore] Using in-memory store (test mode)")
        else:
            _plaid_credential_store = SQLitePlaidCredentialStore()
            logger.info("[PlaidStore] Using SQLite store")

    return _plaid_credential_store


def reset_plaid_credential_store() -> None:
    """Reset the global store (for testing)."""
    global _plaid_credential_store
    _plaid_credential_store = None


__all__ = [
    "PlaidCredentialStore",
    "SQLitePlaidCredentialStore",
    "InMemoryPlaidCredentialStore",
    "get_plaid_credential_store",
    "reset_plaid_credential_store",
]
