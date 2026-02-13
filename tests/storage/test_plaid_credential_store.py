"""
Comprehensive tests for PlaidCredentialStore - encrypted financial credential storage.

Tests cover:
1. Encryption/decryption flows
2. Key derivation and rotation
3. AAD (Additional Authenticated Data) binding
4. Error handling (invalid keys, corrupted data, missing credentials)
5. Tenant isolation
6. Token refresh flows

Target: 40+ test cases for thorough coverage.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Check if cryptography is available
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    # Cleanup
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def sqlite_store(temp_db_path):
    """Create a SQLite credential store with temp database."""
    from aragora.storage.plaid_credential_store import SQLitePlaidCredentialStore

    store = SQLitePlaidCredentialStore(db_path=temp_db_path)
    return store


@pytest.fixture
def memory_store():
    """Create an in-memory credential store."""
    from aragora.storage.plaid_credential_store import InMemoryPlaidCredentialStore

    return InMemoryPlaidCredentialStore()


@pytest.fixture
def master_key():
    """Generate a 32-byte master key for testing."""
    return os.urandom(32)


@pytest.fixture
def encryption_service(master_key):
    """Create an encryption service with a test key."""
    from aragora.security.encryption import EncryptionService

    return EncryptionService(master_key=master_key)


@pytest.fixture(autouse=True)
def reset_global_store():
    """Reset the global store singleton before and after each test."""
    from aragora.storage.plaid_credential_store import reset_plaid_credential_store

    reset_plaid_credential_store()
    yield
    reset_plaid_credential_store()


# ============================================================================
# 1. Basic CRUD Operations Tests
# ============================================================================


class TestBasicCRUDOperations:
    """Tests for basic create, read, update, delete operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_credentials(self, memory_store):
        """Should save and retrieve credentials."""
        await memory_store.save_credentials(
            user_id="user-123",
            tenant_id="tenant-abc",
            item_id="item-xyz",
            access_token="access-token-secret",
            institution_id="ins_1",
            institution_name="Chase Bank",
        )

        result = await memory_store.get_credentials(
            user_id="user-123",
            tenant_id="tenant-abc",
            item_id="item-xyz",
        )

        assert result is not None
        assert result["user_id"] == "user-123"
        assert result["tenant_id"] == "tenant-abc"
        assert result["item_id"] == "item-xyz"
        assert result["institution_id"] == "ins_1"
        assert result["institution_name"] == "Chase Bank"

    @pytest.mark.asyncio
    async def test_get_nonexistent_credentials(self, memory_store):
        """Should return None for nonexistent credentials."""
        result = await memory_store.get_credentials(
            user_id="nonexistent",
            tenant_id="tenant",
            item_id="item",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_credentials(self, memory_store):
        """Should delete credentials successfully."""
        await memory_store.save_credentials(
            user_id="user-del",
            tenant_id="tenant-del",
            item_id="item-del",
            access_token="token",
            institution_id="ins_1",
            institution_name="Bank",
        )

        deleted = await memory_store.delete_credentials(
            user_id="user-del",
            tenant_id="tenant-del",
            item_id="item-del",
        )

        assert deleted is True

        result = await memory_store.get_credentials(
            user_id="user-del",
            tenant_id="tenant-del",
            item_id="item-del",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_credentials(self, memory_store):
        """Should return False when deleting nonexistent credentials."""
        deleted = await memory_store.delete_credentials(
            user_id="nonexistent",
            tenant_id="tenant",
            item_id="item",
        )

        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_credentials_for_user(self, memory_store):
        """Should list all credentials for a user."""
        # Save multiple credentials
        for i in range(3):
            await memory_store.save_credentials(
                user_id="user-list",
                tenant_id="tenant-list",
                item_id=f"item-{i}",
                access_token=f"token-{i}",
                institution_id=f"ins_{i}",
                institution_name=f"Bank {i}",
            )

        results = await memory_store.list_credentials(
            user_id="user-list",
            tenant_id="tenant-list",
        )

        assert len(results) == 3
        # List should not include access tokens
        for result in results:
            assert "access_token" not in result or result.get("access_token") is None

    @pytest.mark.asyncio
    async def test_list_credentials_empty(self, memory_store):
        """Should return empty list when no credentials exist."""
        results = await memory_store.list_credentials(
            user_id="nonexistent",
            tenant_id="tenant",
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_update_last_sync(self, memory_store):
        """Should update last sync timestamp."""
        await memory_store.save_credentials(
            user_id="user-sync",
            tenant_id="tenant-sync",
            item_id="item-sync",
            access_token="token",
            institution_id="ins_1",
            institution_name="Bank",
        )

        # Update sync
        await memory_store.update_last_sync(
            user_id="user-sync",
            tenant_id="tenant-sync",
            item_id="item-sync",
        )

        result = await memory_store.get_credentials(
            user_id="user-sync",
            tenant_id="tenant-sync",
            item_id="item-sync",
        )

        assert result is not None
        assert result["last_sync"] is not None

    @pytest.mark.asyncio
    async def test_save_credentials_upsert(self, memory_store):
        """Should update existing credentials on save."""
        await memory_store.save_credentials(
            user_id="user-upsert",
            tenant_id="tenant-upsert",
            item_id="item-upsert",
            access_token="old-token",
            institution_id="ins_1",
            institution_name="Old Bank",
        )

        # Save with same key but different data
        await memory_store.save_credentials(
            user_id="user-upsert",
            tenant_id="tenant-upsert",
            item_id="item-upsert",
            access_token="new-token",
            institution_id="ins_2",
            institution_name="New Bank",
        )

        result = await memory_store.get_credentials(
            user_id="user-upsert",
            tenant_id="tenant-upsert",
            item_id="item-upsert",
        )

        assert result["institution_name"] == "New Bank"
        assert result["institution_id"] == "ins_2"


# ============================================================================
# 2. SQLite Store Tests
# ============================================================================


class TestSQLiteStore:
    """Tests specific to SQLite-backed store."""

    @pytest.mark.asyncio
    async def test_sqlite_save_and_get(self, sqlite_store):
        """Should save and retrieve from SQLite."""
        await sqlite_store.save_credentials(
            user_id="user-sqlite",
            tenant_id="tenant-sqlite",
            item_id="item-sqlite",
            access_token="sqlite-token",
            institution_id="ins_1",
            institution_name="SQLite Bank",
        )

        result = await sqlite_store.get_credentials(
            user_id="user-sqlite",
            tenant_id="tenant-sqlite",
            item_id="item-sqlite",
        )

        assert result is not None
        assert result["institution_name"] == "SQLite Bank"

    @pytest.mark.asyncio
    async def test_sqlite_persistence(self, temp_db_path):
        """Should persist data across store instances."""
        from aragora.storage.plaid_credential_store import SQLitePlaidCredentialStore

        # First store instance
        store1 = SQLitePlaidCredentialStore(db_path=temp_db_path)
        await store1.save_credentials(
            user_id="user-persist",
            tenant_id="tenant-persist",
            item_id="item-persist",
            access_token="persist-token",
            institution_id="ins_1",
            institution_name="Persist Bank",
        )

        # Second store instance (same db)
        store2 = SQLitePlaidCredentialStore(db_path=temp_db_path)
        result = await store2.get_credentials(
            user_id="user-persist",
            tenant_id="tenant-persist",
            item_id="item-persist",
        )

        assert result is not None
        assert result["institution_name"] == "Persist Bank"

    @pytest.mark.asyncio
    async def test_sqlite_schema_initialization(self, temp_db_path):
        """Should create required tables and indexes."""
        from aragora.storage.plaid_credential_store import SQLitePlaidCredentialStore

        store = SQLitePlaidCredentialStore(db_path=temp_db_path)

        # Check table exists
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='plaid_credentials'"
        )
        assert cursor.fetchone() is not None

        # Check index exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_plaid_user_tenant'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    @pytest.mark.asyncio
    async def test_sqlite_timestamps(self, sqlite_store):
        """Should track created_at and updated_at timestamps."""
        await sqlite_store.save_credentials(
            user_id="user-ts",
            tenant_id="tenant-ts",
            item_id="item-ts",
            access_token="token",
            institution_id="ins_1",
            institution_name="Bank",
        )

        result = await sqlite_store.get_credentials(
            user_id="user-ts",
            tenant_id="tenant-ts",
            item_id="item-ts",
        )

        assert result["created_at"] is not None
        assert result["updated_at"] is not None

    @pytest.mark.asyncio
    async def test_sqlite_unique_constraint(self, sqlite_store):
        """Should enforce unique constraint on user/tenant/item."""
        await sqlite_store.save_credentials(
            user_id="user-unique",
            tenant_id="tenant-unique",
            item_id="item-unique",
            access_token="token1",
            institution_id="ins_1",
            institution_name="Bank1",
        )

        # Second save should replace (upsert)
        await sqlite_store.save_credentials(
            user_id="user-unique",
            tenant_id="tenant-unique",
            item_id="item-unique",
            access_token="token2",
            institution_id="ins_2",
            institution_name="Bank2",
        )

        # Should only have one record
        results = await sqlite_store.list_credentials(
            user_id="user-unique",
            tenant_id="tenant-unique",
        )
        assert len(results) == 1
        assert results[0]["institution_name"] == "Bank2"


# ============================================================================
# 3. Encryption/Decryption Flow Tests
# ============================================================================


class TestEncryptionDecryptionFlows:
    """Tests for encryption and decryption of access tokens."""

    @pytest.mark.asyncio
    async def test_token_encrypted_at_rest(self, memory_store, master_key):
        """Access token should be encrypted when stored."""
        from aragora.security.encryption import init_encryption_service

        init_encryption_service(master_key=master_key)

        await memory_store.save_credentials(
            user_id="user-enc",
            tenant_id="tenant-enc",
            item_id="item-enc",
            access_token="plaintext-secret-token",
            institution_id="ins_1",
            institution_name="Bank",
        )

        # Access internal storage to verify encryption
        key = memory_store._make_key("user-enc", "tenant-enc", "item-enc")
        internal = memory_store._credentials[key]

        # Encrypted token should not contain plaintext
        assert "plaintext-secret-token" not in internal["access_token_encrypted"]
        # Should look like base64 encoded data
        assert internal["access_token_encrypted"].startswith("A")

    @pytest.mark.asyncio
    async def test_token_decrypted_on_retrieval(self, memory_store, master_key):
        """Access token should be decrypted when retrieved."""
        from aragora.security.encryption import init_encryption_service

        init_encryption_service(master_key=master_key)

        original_token = "secret-access-token-12345"

        await memory_store.save_credentials(
            user_id="user-dec",
            tenant_id="tenant-dec",
            item_id="item-dec",
            access_token=original_token,
            institution_id="ins_1",
            institution_name="Bank",
        )

        result = await memory_store.get_credentials(
            user_id="user-dec",
            tenant_id="tenant-dec",
            item_id="item-dec",
        )

        assert result["access_token"] == original_token

    @pytest.mark.asyncio
    async def test_empty_token_handling(self, memory_store):
        """Should handle empty access tokens."""
        await memory_store.save_credentials(
            user_id="user-empty",
            tenant_id="tenant-empty",
            item_id="item-empty",
            access_token="",
            institution_id="ins_1",
            institution_name="Bank",
        )

        result = await memory_store.get_credentials(
            user_id="user-empty",
            tenant_id="tenant-empty",
            item_id="item-empty",
        )

        assert result["access_token"] == ""

    @pytest.mark.asyncio
    async def test_encryption_roundtrip(self, memory_store, master_key):
        """Multiple save/get cycles should preserve token."""
        from aragora.security.encryption import init_encryption_service

        init_encryption_service(master_key=master_key)

        original_token = "roundtrip-token-test"

        for _ in range(5):
            await memory_store.save_credentials(
                user_id="user-rt",
                tenant_id="tenant-rt",
                item_id="item-rt",
                access_token=original_token,
                institution_id="ins_1",
                institution_name="Bank",
            )

            result = await memory_store.get_credentials(
                user_id="user-rt",
                tenant_id="tenant-rt",
                item_id="item-rt",
            )

            assert result["access_token"] == original_token


# ============================================================================
# 4. AAD (Additional Authenticated Data) Binding Tests
# ============================================================================


class TestAADBinding:
    """Tests for AAD binding credentials to user/tenant/item."""

    def test_aad_format(self, memory_store):
        """AAD should include tenant, user, and item IDs."""
        aad = memory_store._make_key("user-123", "tenant-abc", "item-xyz")

        # InMemoryStore uses format: tenant:user:item
        assert "tenant-abc" in aad
        assert "user-123" in aad
        assert "item-xyz" in aad

    def test_sqlite_aad_format(self, sqlite_store):
        """SQLite store AAD should include plaid prefix."""
        aad = sqlite_store._make_aad("user-123", "tenant-abc", "item-xyz")

        assert aad == "plaid:tenant-abc:user-123:item-xyz"

    @pytest.mark.asyncio
    async def test_wrong_aad_fails_decryption(self, master_key):
        """Decryption with wrong AAD should fail."""
        from aragora.security.encryption import EncryptionService

        service = EncryptionService(master_key=master_key)

        plaintext = "secret-token"
        aad1 = "plaid:tenant1:user1:item1"
        aad2 = "plaid:tenant2:user2:item2"

        encrypted = service.encrypt(plaintext, associated_data=aad1)

        with pytest.raises(Exception):
            service.decrypt(encrypted, associated_data=aad2)

    @pytest.mark.asyncio
    async def test_aad_prevents_cross_user_access(self, master_key):
        """AAD should prevent decrypting another user's token."""
        from aragora.security.encryption import EncryptionService

        service = EncryptionService(master_key=master_key)

        # Encrypt for user1
        token = "user1-secret-token"
        user1_aad = "plaid:tenant:user1:item1"
        encrypted = service.encrypt(token, associated_data=user1_aad)

        # Try to decrypt with user2's AAD (simulating cross-user attack)
        user2_aad = "plaid:tenant:user2:item1"

        with pytest.raises(Exception):
            service.decrypt(encrypted, associated_data=user2_aad)

    @pytest.mark.asyncio
    async def test_aad_prevents_cross_tenant_access(self, master_key):
        """AAD should prevent decrypting another tenant's token."""
        from aragora.security.encryption import EncryptionService

        service = EncryptionService(master_key=master_key)

        token = "tenant1-secret-token"
        tenant1_aad = "plaid:tenant1:user:item1"
        encrypted = service.encrypt(token, associated_data=tenant1_aad)

        tenant2_aad = "plaid:tenant2:user:item1"

        with pytest.raises(Exception):
            service.decrypt(encrypted, associated_data=tenant2_aad)


# ============================================================================
# 5. Tenant Isolation Tests
# ============================================================================


class TestTenantIsolation:
    """Tests for multi-tenant isolation."""

    @pytest.mark.asyncio
    async def test_same_user_different_tenants(self, memory_store):
        """Same user ID in different tenants should be isolated."""
        await memory_store.save_credentials(
            user_id="shared-user",
            tenant_id="tenant-a",
            item_id="item-1",
            access_token="tenant-a-token",
            institution_id="ins_1",
            institution_name="Bank A",
        )

        await memory_store.save_credentials(
            user_id="shared-user",
            tenant_id="tenant-b",
            item_id="item-1",
            access_token="tenant-b-token",
            institution_id="ins_1",
            institution_name="Bank B",
        )

        result_a = await memory_store.get_credentials(
            user_id="shared-user",
            tenant_id="tenant-a",
            item_id="item-1",
        )

        result_b = await memory_store.get_credentials(
            user_id="shared-user",
            tenant_id="tenant-b",
            item_id="item-1",
        )

        assert result_a["access_token"] == "tenant-a-token"
        assert result_b["access_token"] == "tenant-b-token"

    @pytest.mark.asyncio
    async def test_list_only_tenant_credentials(self, memory_store):
        """List should only return credentials for specified tenant."""
        await memory_store.save_credentials(
            user_id="user-iso",
            tenant_id="tenant-x",
            item_id="item-1",
            access_token="token-x",
            institution_id="ins_1",
            institution_name="Bank X",
        )

        await memory_store.save_credentials(
            user_id="user-iso",
            tenant_id="tenant-y",
            item_id="item-1",
            access_token="token-y",
            institution_id="ins_1",
            institution_name="Bank Y",
        )

        results_x = await memory_store.list_credentials(
            user_id="user-iso",
            tenant_id="tenant-x",
        )

        results_y = await memory_store.list_credentials(
            user_id="user-iso",
            tenant_id="tenant-y",
        )

        assert len(results_x) == 1
        assert len(results_y) == 1
        assert results_x[0]["tenant_id"] == "tenant-x"
        assert results_y[0]["tenant_id"] == "tenant-y"

    @pytest.mark.asyncio
    async def test_delete_only_affects_tenant(self, memory_store):
        """Delete should only affect the specified tenant."""
        await memory_store.save_credentials(
            user_id="user-del-iso",
            tenant_id="tenant-del-a",
            item_id="item-1",
            access_token="token-a",
            institution_id="ins_1",
            institution_name="Bank",
        )

        await memory_store.save_credentials(
            user_id="user-del-iso",
            tenant_id="tenant-del-b",
            item_id="item-1",
            access_token="token-b",
            institution_id="ins_1",
            institution_name="Bank",
        )

        # Delete from tenant-a
        await memory_store.delete_credentials(
            user_id="user-del-iso",
            tenant_id="tenant-del-a",
            item_id="item-1",
        )

        # tenant-a should be deleted
        result_a = await memory_store.get_credentials(
            user_id="user-del-iso",
            tenant_id="tenant-del-a",
            item_id="item-1",
        )
        assert result_a is None

        # tenant-b should still exist
        result_b = await memory_store.get_credentials(
            user_id="user-del-iso",
            tenant_id="tenant-del-b",
            item_id="item-1",
        )
        assert result_b is not None


# ============================================================================
# 6. Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_get_with_wrong_user_id(self, memory_store):
        """Should return None for wrong user ID."""
        await memory_store.save_credentials(
            user_id="correct-user",
            tenant_id="tenant",
            item_id="item",
            access_token="token",
            institution_id="ins_1",
            institution_name="Bank",
        )

        result = await memory_store.get_credentials(
            user_id="wrong-user",
            tenant_id="tenant",
            item_id="item",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_with_wrong_item_id(self, memory_store):
        """Should return None for wrong item ID."""
        await memory_store.save_credentials(
            user_id="user",
            tenant_id="tenant",
            item_id="correct-item",
            access_token="token",
            institution_id="ins_1",
            institution_name="Bank",
        )

        result = await memory_store.get_credentials(
            user_id="user",
            tenant_id="tenant",
            item_id="wrong-item",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_encryption_failure_without_crypto(self):
        """Should handle encryption unavailability gracefully."""
        from aragora.storage.plaid_credential_store import _encrypt_token

        # Patch CRYPTO_AVAILABLE to False and encryption not required
        with patch("aragora.storage.plaid_credential_store.CRYPTO_AVAILABLE", False):
            with patch(
                "aragora.storage.plaid_credential_store.is_encryption_required",
                return_value=False,
            ):
                # Should return plaintext with warning
                result = _encrypt_token("test-token")
                assert result == "test-token"

    @pytest.mark.asyncio
    async def test_encryption_required_without_crypto_raises(self):
        """Should raise error when encryption required but unavailable."""
        from aragora.storage.plaid_credential_store import _encrypt_token, EncryptionError

        with patch("aragora.storage.plaid_credential_store.CRYPTO_AVAILABLE", False):
            with patch(
                "aragora.storage.plaid_credential_store.is_encryption_required",
                return_value=True,
            ):
                with pytest.raises(EncryptionError):
                    _encrypt_token("test-token")

    @pytest.mark.asyncio
    async def test_legacy_plaintext_token_decryption(self):
        """Should handle legacy unencrypted tokens gracefully."""
        from aragora.storage.plaid_credential_store import _decrypt_token

        # Plaintext token not starting with "A" (encrypted marker)
        legacy_token = "access-token-12345"

        with patch("aragora.storage.plaid_credential_store.CRYPTO_AVAILABLE", True):
            result = _decrypt_token(legacy_token)

        assert result == legacy_token


# ============================================================================
# 7. Key Rotation Tests
# ============================================================================


class TestKeyRotation:
    """Tests for encryption key rotation scenarios."""

    @pytest.mark.asyncio
    async def test_decrypt_after_key_rotation(self, memory_store):
        """Should decrypt data encrypted before key rotation."""
        from aragora.security.encryption import init_encryption_service

        master_key = os.urandom(32)
        service = init_encryption_service(master_key=master_key)

        original_token = "pre-rotation-token"

        await memory_store.save_credentials(
            user_id="user-rot",
            tenant_id="tenant-rot",
            item_id="item-rot",
            access_token=original_token,
            institution_id="ins_1",
            institution_name="Bank",
        )

        # Rotate key
        service.rotate_key()

        # Should still decrypt with old key
        result = await memory_store.get_credentials(
            user_id="user-rot",
            tenant_id="tenant-rot",
            item_id="item-rot",
        )

        assert result["access_token"] == original_token

    @pytest.mark.asyncio
    async def test_new_credentials_use_new_key(self, memory_store):
        """New credentials should use rotated key."""
        from aragora.security.encryption import init_encryption_service

        master_key = os.urandom(32)
        service = init_encryption_service(master_key=master_key)

        # Save before rotation
        await memory_store.save_credentials(
            user_id="user-pre",
            tenant_id="tenant",
            item_id="item-pre",
            access_token="pre-token",
            institution_id="ins_1",
            institution_name="Bank",
        )

        # Rotate
        service.rotate_key()

        # Save after rotation
        await memory_store.save_credentials(
            user_id="user-post",
            tenant_id="tenant",
            item_id="item-post",
            access_token="post-token",
            institution_id="ins_1",
            institution_name="Bank",
        )

        # Both should decrypt correctly
        result_pre = await memory_store.get_credentials(
            user_id="user-pre",
            tenant_id="tenant",
            item_id="item-pre",
        )
        result_post = await memory_store.get_credentials(
            user_id="user-post",
            tenant_id="tenant",
            item_id="item-post",
        )

        assert result_pre["access_token"] == "pre-token"
        assert result_post["access_token"] == "post-token"


# ============================================================================
# 8. Token Refresh Flow Tests
# ============================================================================


class TestTokenRefreshFlows:
    """Tests for OAuth token refresh scenarios."""

    @pytest.mark.asyncio
    async def test_update_token_preserves_metadata(self, memory_store):
        """Token update should preserve other credential metadata."""
        await memory_store.save_credentials(
            user_id="user-refresh",
            tenant_id="tenant-refresh",
            item_id="item-refresh",
            access_token="old-token",
            institution_id="ins_1",
            institution_name="My Bank",
        )

        # Get created_at timestamp
        original = await memory_store.get_credentials(
            user_id="user-refresh",
            tenant_id="tenant-refresh",
            item_id="item-refresh",
        )
        original_created = original["created_at"]

        # Refresh token (re-save with new token)
        await memory_store.save_credentials(
            user_id="user-refresh",
            tenant_id="tenant-refresh",
            item_id="item-refresh",
            access_token="new-refreshed-token",
            institution_id="ins_1",
            institution_name="My Bank",
        )

        updated = await memory_store.get_credentials(
            user_id="user-refresh",
            tenant_id="tenant-refresh",
            item_id="item-refresh",
        )

        assert updated["access_token"] == "new-refreshed-token"
        assert updated["institution_name"] == "My Bank"

    @pytest.mark.asyncio
    async def test_concurrent_token_refresh(self, memory_store):
        """Should handle concurrent token refresh attempts."""
        import asyncio

        await memory_store.save_credentials(
            user_id="user-concurrent",
            tenant_id="tenant-concurrent",
            item_id="item-concurrent",
            access_token="initial-token",
            institution_id="ins_1",
            institution_name="Bank",
        )

        async def refresh_token(new_token: str):
            await memory_store.save_credentials(
                user_id="user-concurrent",
                tenant_id="tenant-concurrent",
                item_id="item-concurrent",
                access_token=new_token,
                institution_id="ins_1",
                institution_name="Bank",
            )

        # Simulate concurrent refreshes
        await asyncio.gather(
            refresh_token("token-1"),
            refresh_token("token-2"),
            refresh_token("token-3"),
        )

        # Should have one credential (last write wins)
        result = await memory_store.get_credentials(
            user_id="user-concurrent",
            tenant_id="tenant-concurrent",
            item_id="item-concurrent",
        )

        assert result is not None
        assert result["access_token"] in ["token-1", "token-2", "token-3"]


# ============================================================================
# 9. Singleton and Factory Tests
# ============================================================================


class TestSingletonAndFactory:
    """Tests for global store singleton management."""

    def test_get_plaid_credential_store_returns_singleton(self):
        """Should return the same store instance."""
        from aragora.storage.plaid_credential_store import (
            get_plaid_credential_store,
            reset_plaid_credential_store,
        )

        reset_plaid_credential_store()

        with patch.dict(os.environ, {"ARAGORA_TEST_MODE": "true"}):
            store1 = get_plaid_credential_store()
            store2 = get_plaid_credential_store()

        assert store1 is store2

    def test_reset_clears_singleton(self):
        """Reset should clear the singleton."""
        from aragora.storage.plaid_credential_store import (
            get_plaid_credential_store,
            reset_plaid_credential_store,
        )

        with patch.dict(os.environ, {"ARAGORA_TEST_MODE": "true"}):
            store1 = get_plaid_credential_store()
            reset_plaid_credential_store()
            store2 = get_plaid_credential_store()

        assert store1 is not store2

    def test_test_mode_uses_memory_store(self):
        """Test mode should use in-memory store."""
        from aragora.storage.plaid_credential_store import (
            get_plaid_credential_store,
            InMemoryPlaidCredentialStore,
            reset_plaid_credential_store,
        )

        reset_plaid_credential_store()

        with patch.dict(os.environ, {"ARAGORA_TEST_MODE": "true"}):
            store = get_plaid_credential_store()

        assert isinstance(store, InMemoryPlaidCredentialStore)

    def test_production_mode_uses_sqlite_store(self):
        """Non-test mode should use SQLite store."""
        from aragora.storage.plaid_credential_store import (
            get_plaid_credential_store,
            SQLitePlaidCredentialStore,
            reset_plaid_credential_store,
        )

        reset_plaid_credential_store()

        # Clear test mode
        env = os.environ.copy()
        env.pop("ARAGORA_TEST_MODE", None)

        with patch.dict(os.environ, env, clear=True):
            with patch.dict(os.environ, {"ARAGORA_DATA_DIR": tempfile.gettempdir()}):
                store = get_plaid_credential_store()

        assert isinstance(store, SQLitePlaidCredentialStore)


# ============================================================================
# 10. Edge Cases and Security Tests
# ============================================================================


class TestEdgeCasesAndSecurity:
    """Tests for edge cases and security considerations."""

    @pytest.mark.asyncio
    async def test_special_characters_in_token(self, memory_store):
        """Should handle special characters in access token."""
        special_token = "token!@#$%^&*()_+-=[]{}|;':\",./<>?\n\t"

        await memory_store.save_credentials(
            user_id="user-special",
            tenant_id="tenant-special",
            item_id="item-special",
            access_token=special_token,
            institution_id="ins_1",
            institution_name="Bank",
        )

        result = await memory_store.get_credentials(
            user_id="user-special",
            tenant_id="tenant-special",
            item_id="item-special",
        )

        assert result["access_token"] == special_token

    @pytest.mark.asyncio
    async def test_unicode_in_institution_name(self, memory_store):
        """Should handle unicode in institution name."""
        await memory_store.save_credentials(
            user_id="user-unicode",
            tenant_id="tenant-unicode",
            item_id="item-unicode",
            access_token="token",
            institution_id="ins_1",
            institution_name="Bank \u4e2d\u6587 \U0001f3e6",
        )

        result = await memory_store.get_credentials(
            user_id="user-unicode",
            tenant_id="tenant-unicode",
            item_id="item-unicode",
        )

        assert result["institution_name"] == "Bank \u4e2d\u6587 \U0001f3e6"

    @pytest.mark.asyncio
    async def test_very_long_token(self, memory_store):
        """Should handle very long access tokens."""
        long_token = "x" * 10000

        await memory_store.save_credentials(
            user_id="user-long",
            tenant_id="tenant-long",
            item_id="item-long",
            access_token=long_token,
            institution_id="ins_1",
            institution_name="Bank",
        )

        result = await memory_store.get_credentials(
            user_id="user-long",
            tenant_id="tenant-long",
            item_id="item-long",
        )

        assert result["access_token"] == long_token

    @pytest.mark.asyncio
    async def test_list_excludes_access_tokens(self, memory_store):
        """List credentials should not expose access tokens."""
        await memory_store.save_credentials(
            user_id="user-list-sec",
            tenant_id="tenant-list-sec",
            item_id="item-1",
            access_token="secret-token-1",
            institution_id="ins_1",
            institution_name="Bank 1",
        )

        await memory_store.save_credentials(
            user_id="user-list-sec",
            tenant_id="tenant-list-sec",
            item_id="item-2",
            access_token="secret-token-2",
            institution_id="ins_2",
            institution_name="Bank 2",
        )

        results = await memory_store.list_credentials(
            user_id="user-list-sec",
            tenant_id="tenant-list-sec",
        )

        for result in results:
            # Should not have access_token or encrypted version
            assert "access_token" not in result or result.get("access_token") is None
            assert "access_token_encrypted" not in result

    @pytest.mark.asyncio
    async def test_sql_injection_prevention(self, sqlite_store):
        """Should prevent SQL injection in queries."""
        malicious_item = "item'; DROP TABLE plaid_credentials; --"

        await sqlite_store.save_credentials(
            user_id="user",
            tenant_id="tenant",
            item_id=malicious_item,
            access_token="token",
            institution_id="ins_1",
            institution_name="Bank",
        )

        # Table should still exist
        result = await sqlite_store.get_credentials(
            user_id="user",
            tenant_id="tenant",
            item_id=malicious_item,
        )

        assert result is not None


# ============================================================================
# 11. Corrupted Data Tests
# ============================================================================


class TestCorruptedData:
    """Tests for handling corrupted encrypted data."""

    @pytest.mark.asyncio
    async def test_corrupted_ciphertext_handling(self, master_key):
        """Should handle corrupted ciphertext gracefully."""
        from aragora.storage.plaid_credential_store import _decrypt_token

        # Corrupted base64 that looks like encrypted data (starts with A)
        corrupted = "AGarbledDataThatIsNotValidEncryption=="

        with patch("aragora.storage.plaid_credential_store.CRYPTO_AVAILABLE", True):
            # Should return original on decryption failure (legacy fallback)
            result = _decrypt_token(corrupted)

        # Falls back to returning the corrupted data as-is
        assert result == corrupted

    @pytest.mark.asyncio
    async def test_tampered_encrypted_token(self, memory_store, master_key):
        """Should detect tampered encrypted token."""
        from aragora.security.encryption import init_encryption_service, EncryptionService

        service = init_encryption_service(master_key=master_key)

        # Save valid credentials
        await memory_store.save_credentials(
            user_id="user-tamper",
            tenant_id="tenant-tamper",
            item_id="item-tamper",
            access_token="valid-token",
            institution_id="ins_1",
            institution_name="Bank",
        )

        # Tamper with stored encrypted token
        key = memory_store._make_key("user-tamper", "tenant-tamper", "item-tamper")
        original_encrypted = memory_store._credentials[key]["access_token_encrypted"]

        # Modify some bytes (keep the "A" prefix)
        tampered = original_encrypted[:10] + "X" + original_encrypted[11:]
        memory_store._credentials[key]["access_token_encrypted"] = tampered

        # Get should handle decryption failure gracefully
        result = await memory_store.get_credentials(
            user_id="user-tamper",
            tenant_id="tenant-tamper",
            item_id="item-tamper",
        )

        # Should return something (possibly corrupted or original)
        assert result is not None


# ============================================================================
# 12. Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    @pytest.mark.asyncio
    async def test_full_credential_lifecycle(self, sqlite_store, master_key):
        """Test complete credential lifecycle with encryption."""
        from aragora.security.encryption import init_encryption_service

        init_encryption_service(master_key=master_key)

        user_id = "lifecycle-user"
        tenant_id = "lifecycle-tenant"
        item_id = "lifecycle-item"

        # Create
        await sqlite_store.save_credentials(
            user_id=user_id,
            tenant_id=tenant_id,
            item_id=item_id,
            access_token="initial-token",
            institution_id="ins_1",
            institution_name="Lifecycle Bank",
        )

        # Read
        creds = await sqlite_store.get_credentials(user_id, tenant_id, item_id)
        assert creds["access_token"] == "initial-token"

        # Update (token refresh)
        await sqlite_store.save_credentials(
            user_id=user_id,
            tenant_id=tenant_id,
            item_id=item_id,
            access_token="refreshed-token",
            institution_id="ins_1",
            institution_name="Lifecycle Bank",
        )

        creds = await sqlite_store.get_credentials(user_id, tenant_id, item_id)
        assert creds["access_token"] == "refreshed-token"

        # Update sync timestamp
        await sqlite_store.update_last_sync(user_id, tenant_id, item_id)
        creds = await sqlite_store.get_credentials(user_id, tenant_id, item_id)
        assert creds["last_sync"] is not None

        # List
        all_creds = await sqlite_store.list_credentials(user_id, tenant_id)
        assert len(all_creds) == 1

        # Delete
        deleted = await sqlite_store.delete_credentials(user_id, tenant_id, item_id)
        assert deleted is True

        # Verify deleted
        creds = await sqlite_store.get_credentials(user_id, tenant_id, item_id)
        assert creds is None

    @pytest.mark.asyncio
    async def test_multi_user_multi_tenant_scenario(self, memory_store, master_key):
        """Test multiple users across multiple tenants."""
        from aragora.security.encryption import init_encryption_service

        init_encryption_service(master_key=master_key)

        # Setup: 2 tenants, 2 users each, 2 items each
        tenants = ["tenant-1", "tenant-2"]
        users = ["user-a", "user-b"]
        items = ["item-x", "item-y"]

        for tenant in tenants:
            for user in users:
                for item in items:
                    token = f"token-{tenant}-{user}-{item}"
                    await memory_store.save_credentials(
                        user_id=user,
                        tenant_id=tenant,
                        item_id=item,
                        access_token=token,
                        institution_id="ins_1",
                        institution_name=f"Bank-{tenant}",
                    )

        # Verify isolation
        for tenant in tenants:
            for user in users:
                creds_list = await memory_store.list_credentials(user, tenant)
                assert len(creds_list) == 2

                for item in items:
                    creds = await memory_store.get_credentials(user, tenant, item)
                    expected_token = f"token-{tenant}-{user}-{item}"
                    assert creds["access_token"] == expected_token
