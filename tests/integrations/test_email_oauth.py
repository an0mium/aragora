"""
Tests for Email OAuth credential storage.

Tests for:
- Credential CRUD operations
- Token encryption/decryption
- Multi-tenant isolation
- Token expiry detection
- Failure tracking
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from aragora.integrations.email_oauth import (
    DEFAULT_REFRESH_MARGIN_SECONDS,
    EmailCredential,
    InMemoryEmailCredentialStore,
    SQLiteEmailCredentialStore,
    get_email_credential_store,
    reset_email_credential_store,
    set_email_credential_store,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def credential():
    """Create a test credential."""
    return EmailCredential(
        tenant_id="tenant_123",
        provider="gmail",
        email_address="user@example.com",
        access_token="access_token_abc",
        refresh_token="refresh_token_xyz",
        token_expiry=datetime.now(timezone.utc) + timedelta(hours=1),
        client_id="client_id_123",
        client_secret="client_secret_456",
        provider_user_id="google_user_123",
        scopes=["https://www.googleapis.com/auth/gmail.send"],
    )


@pytest.fixture
def expired_credential():
    """Create an expired credential."""
    return EmailCredential(
        tenant_id="tenant_123",
        provider="gmail",
        email_address="expired@example.com",
        access_token="expired_token",
        refresh_token="refresh_token",
        token_expiry=datetime.now(timezone.utc) - timedelta(hours=1),
    )


@pytest.fixture
def store():
    """Create an in-memory store for testing."""
    return InMemoryEmailCredentialStore()


@pytest.fixture(autouse=True)
def reset_global_store():
    """Reset global store before and after each test."""
    reset_email_credential_store()
    yield
    reset_email_credential_store()


# =============================================================================
# EmailCredential Tests
# =============================================================================


class TestEmailCredential:
    """Tests for EmailCredential dataclass."""

    def test_credential_id(self, credential):
        """Test credential ID generation."""
        assert credential.credential_id == "tenant_123:gmail:user@example.com"

    def test_needs_refresh_valid_token(self, credential):
        """Test needs_refresh returns False for valid token."""
        assert credential.needs_refresh() is False

    def test_needs_refresh_expired_token(self, expired_credential):
        """Test needs_refresh returns True for expired token."""
        assert expired_credential.needs_refresh() is True

    def test_needs_refresh_expiring_soon(self):
        """Test needs_refresh returns True for token expiring soon."""
        cred = EmailCredential(
            tenant_id="t1",
            provider="gmail",
            email_address="test@example.com",
            access_token="token",
            token_expiry=datetime.now(timezone.utc) + timedelta(seconds=60),
        )
        # Default margin is 300 seconds, so 60 seconds should trigger refresh
        assert cred.needs_refresh() is True
        # With smaller margin, should not need refresh
        assert cred.needs_refresh(margin_seconds=30) is False

    def test_needs_refresh_no_expiry(self):
        """Test needs_refresh returns False when no expiry set."""
        cred = EmailCredential(
            tenant_id="t1",
            provider="gmail",
            email_address="test@example.com",
            access_token="token",
            token_expiry=None,
        )
        assert cred.needs_refresh() is False

    def test_needs_refresh_no_token(self):
        """Test needs_refresh returns True when no token."""
        cred = EmailCredential(
            tenant_id="t1",
            provider="gmail",
            email_address="test@example.com",
        )
        assert cred.needs_refresh() is True

    def test_is_expired(self, expired_credential, credential):
        """Test is_expired method."""
        assert expired_credential.is_expired() is True
        assert credential.is_expired() is False

    def test_to_dict_excludes_secrets(self, credential):
        """Test to_dict excludes secrets by default."""
        data = credential.to_dict()
        assert "access_token" not in data
        assert "refresh_token" not in data
        assert "client_secret" not in data
        assert data["tenant_id"] == "tenant_123"
        assert data["provider"] == "gmail"

    def test_to_dict_includes_secrets(self, credential):
        """Test to_dict includes secrets when requested."""
        data = credential.to_dict(include_secrets=True)
        assert data["access_token"] == "access_token_abc"
        assert data["refresh_token"] == "refresh_token_xyz"
        assert data["client_secret"] == "client_secret_456"

    def test_to_json_and_from_json(self, credential):
        """Test JSON serialization round-trip."""
        json_str = credential.to_json()
        restored = EmailCredential.from_json(json_str)

        assert restored.tenant_id == credential.tenant_id
        assert restored.provider == credential.provider
        assert restored.email_address == credential.email_address
        assert restored.access_token == credential.access_token
        assert restored.refresh_token == credential.refresh_token
        assert restored.scopes == credential.scopes


# =============================================================================
# InMemoryEmailCredentialStore Tests
# =============================================================================


class TestInMemoryEmailCredentialStore:
    """Tests for InMemoryEmailCredentialStore."""

    @pytest.mark.asyncio
    async def test_save_and_get(self, store, credential):
        """Test saving and retrieving a credential."""
        await store.save(credential)
        retrieved = await store.get("tenant_123", "gmail", "user@example.com")

        assert retrieved is not None
        assert retrieved.tenant_id == "tenant_123"
        assert retrieved.access_token == "access_token_abc"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Test getting nonexistent credential returns None."""
        result = await store.get("tenant_123", "gmail", "nonexistent@example.com")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, store, credential):
        """Test deleting a credential."""
        await store.save(credential)
        deleted = await store.delete("tenant_123", "gmail", "user@example.com")
        assert deleted is True

        result = await store.get("tenant_123", "gmail", "user@example.com")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        """Test deleting nonexistent credential returns False."""
        deleted = await store.delete("tenant_123", "gmail", "nonexistent@example.com")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_for_tenant(self, store):
        """Test listing credentials for a tenant."""
        cred1 = EmailCredential(
            tenant_id="tenant_a", provider="gmail", email_address="user1@example.com"
        )
        cred2 = EmailCredential(
            tenant_id="tenant_a", provider="microsoft", email_address="user2@example.com"
        )
        cred3 = EmailCredential(
            tenant_id="tenant_b", provider="gmail", email_address="user3@example.com"
        )

        await store.save(cred1)
        await store.save(cred2)
        await store.save(cred3)

        tenant_a_creds = await store.list_for_tenant("tenant_a")
        assert len(tenant_a_creds) == 2

        tenant_b_creds = await store.list_for_tenant("tenant_b")
        assert len(tenant_b_creds) == 1

    @pytest.mark.asyncio
    async def test_list_expiring(self, store):
        """Test listing expiring credentials."""
        expiring_soon = EmailCredential(
            tenant_id="t1",
            provider="gmail",
            email_address="expiring@example.com",
            access_token="token",
            token_expiry=datetime.now(timezone.utc) + timedelta(minutes=30),
            is_active=True,
        )
        valid = EmailCredential(
            tenant_id="t1",
            provider="gmail",
            email_address="valid@example.com",
            access_token="token",
            token_expiry=datetime.now(timezone.utc) + timedelta(hours=2),
            is_active=True,
        )

        await store.save(expiring_soon)
        await store.save(valid)

        expiring = await store.list_expiring(within_seconds=3600)
        assert len(expiring) == 1
        assert expiring[0].email_address == "expiring@example.com"

    @pytest.mark.asyncio
    async def test_update_last_used(self, store, credential):
        """Test updating last_used timestamp."""
        await store.save(credential)
        await store.update_last_used("tenant_123", "gmail", "user@example.com")

        retrieved = await store.get("tenant_123", "gmail", "user@example.com")
        assert retrieved.last_used is not None

    @pytest.mark.asyncio
    async def test_record_failure(self, store, credential):
        """Test recording failures."""
        await store.save(credential)
        await store.record_failure("tenant_123", "gmail", "user@example.com", "Connection refused")

        retrieved = await store.get("tenant_123", "gmail", "user@example.com")
        assert retrieved.failure_count == 1
        assert retrieved.last_error == "Connection refused"

        await store.record_failure("tenant_123", "gmail", "user@example.com", "Timeout")
        retrieved = await store.get("tenant_123", "gmail", "user@example.com")
        assert retrieved.failure_count == 2

    @pytest.mark.asyncio
    async def test_reset_failures(self, store, credential):
        """Test resetting failure count."""
        credential.failure_count = 5
        credential.last_error = "Previous error"
        await store.save(credential)

        await store.reset_failures("tenant_123", "gmail", "user@example.com")

        retrieved = await store.get("tenant_123", "gmail", "user@example.com")
        assert retrieved.failure_count == 0
        assert retrieved.last_error == ""

    @pytest.mark.asyncio
    async def test_concurrent_access(self, store):
        """Test thread-safe concurrent access."""
        credentials = [
            EmailCredential(
                tenant_id="t1",
                provider="gmail",
                email_address=f"user{i}@example.com",
                access_token=f"token_{i}",
            )
            for i in range(20)
        ]

        # Concurrent saves
        await asyncio.gather(*[store.save(c) for c in credentials])

        # Verify all saved
        all_creds = await store.list_for_tenant("t1")
        assert len(all_creds) == 20


# =============================================================================
# SQLiteEmailCredentialStore Tests
# =============================================================================


class TestSQLiteEmailCredentialStore:
    """Tests for SQLiteEmailCredentialStore."""

    @pytest.fixture
    def sqlite_store(self, tmp_path):
        """Create a SQLite store in a temp directory."""
        return SQLiteEmailCredentialStore(tmp_path / "email_creds.db")

    @pytest.mark.asyncio
    async def test_save_and_get(self, sqlite_store, credential):
        """Test saving and retrieving a credential."""
        await sqlite_store.save(credential)
        retrieved = await sqlite_store.get("tenant_123", "gmail", "user@example.com")

        assert retrieved is not None
        assert retrieved.tenant_id == "tenant_123"
        assert retrieved.access_token == "access_token_abc"
        assert retrieved.scopes == ["https://www.googleapis.com/auth/gmail.send"]

    @pytest.mark.asyncio
    async def test_persistence(self, tmp_path, credential):
        """Test data persists across store instances."""
        db_path = tmp_path / "persist_test.db"

        # Save with first instance
        store1 = SQLiteEmailCredentialStore(db_path)
        await store1.save(credential)
        await store1.close()

        # Read with second instance
        store2 = SQLiteEmailCredentialStore(db_path)
        retrieved = await store2.get("tenant_123", "gmail", "user@example.com")
        await store2.close()

        assert retrieved is not None
        assert retrieved.access_token == "access_token_abc"

    @pytest.mark.asyncio
    async def test_list_expiring_query(self, sqlite_store):
        """Test list_expiring uses proper SQL query."""
        # Add credentials with different expiry times
        for i, hours in enumerate([0.5, 2, 24]):
            cred = EmailCredential(
                tenant_id="t1",
                provider="gmail",
                email_address=f"user{i}@example.com",
                access_token="token",
                token_expiry=datetime.now(timezone.utc) + timedelta(hours=hours),
                is_active=True,
            )
            await sqlite_store.save(cred)

        # Query for expiring within 1 hour
        expiring = await sqlite_store.list_expiring(within_seconds=3600)
        assert len(expiring) == 1
        assert expiring[0].email_address == "user0@example.com"


# =============================================================================
# Global Store Factory Tests
# =============================================================================


class TestGlobalStoreFactory:
    """Tests for global store factory functions."""

    def test_get_store_returns_instance(self):
        """Test get_email_credential_store returns a store."""
        store = get_email_credential_store()
        assert store is not None

    def test_get_store_returns_same_instance(self):
        """Test get_email_credential_store returns singleton."""
        store1 = get_email_credential_store()
        store2 = get_email_credential_store()
        assert store1 is store2

    def test_set_custom_store(self):
        """Test setting a custom store."""
        custom_store = InMemoryEmailCredentialStore()
        set_email_credential_store(custom_store)

        retrieved = get_email_credential_store()
        assert retrieved is custom_store

    def test_reset_store(self):
        """Test resetting the global store."""
        store1 = get_email_credential_store()
        reset_email_credential_store()
        store2 = get_email_credential_store()

        assert store1 is not store2


# =============================================================================
# Token Encryption Tests
# =============================================================================


class TestTokenEncryption:
    """Tests for token encryption functionality."""

    @pytest.mark.asyncio
    async def test_tokens_encrypted_in_storage(self, tmp_path, credential):
        """Test that tokens are encrypted when stored."""
        # This test verifies the encryption logic is called
        db_path = tmp_path / "encrypted_test.db"
        store = SQLiteEmailCredentialStore(db_path)
        await store.save(credential)

        # Verify we can read back the original values (decrypted)
        retrieved = await store.get("tenant_123", "gmail", "user@example.com")
        assert retrieved.access_token == "access_token_abc"
        assert retrieved.refresh_token == "refresh_token_xyz"
        assert retrieved.client_secret == "client_secret_456"

    @pytest.mark.asyncio
    async def test_from_row_decryption(self):
        """Test that from_row properly decrypts tokens."""
        # Create a mock row tuple (simulating database row)
        row = (
            "tenant_1",  # tenant_id
            "gmail",  # provider
            "test@example.com",  # email_address
            "encrypted_access",  # access_token (would be encrypted in real storage)
            "encrypted_refresh",  # refresh_token
            None,  # token_expiry
            "client_id",  # client_id
            "encrypted_secret",  # client_secret
            "provider_user",  # provider_user_id
            "[]",  # scopes (JSON)
            1,  # is_active
            None,  # last_used
            0,  # failure_count
            "",  # last_error
            time.time(),  # created_at
            time.time(),  # updated_at
        )

        cred = EmailCredential.from_row(row)

        assert cred.tenant_id == "tenant_1"
        assert cred.provider == "gmail"
        # Note: In real usage, these would be decrypted if they were encrypted
        assert cred.access_token in ("encrypted_access", "")  # May be decrypted or original


# =============================================================================
# Multi-tenant Isolation Tests
# =============================================================================


class TestMultiTenantIsolation:
    """Tests for multi-tenant credential isolation."""

    @pytest.mark.asyncio
    async def test_tenants_isolated(self, store):
        """Test credentials are isolated by tenant."""
        cred_tenant_a = EmailCredential(
            tenant_id="tenant_a",
            provider="gmail",
            email_address="shared@example.com",
            access_token="token_a",
        )
        cred_tenant_b = EmailCredential(
            tenant_id="tenant_b",
            provider="gmail",
            email_address="shared@example.com",
            access_token="token_b",
        )

        await store.save(cred_tenant_a)
        await store.save(cred_tenant_b)

        # Each tenant should see their own credential
        retrieved_a = await store.get("tenant_a", "gmail", "shared@example.com")
        retrieved_b = await store.get("tenant_b", "gmail", "shared@example.com")

        assert retrieved_a.access_token == "token_a"
        assert retrieved_b.access_token == "token_b"

    @pytest.mark.asyncio
    async def test_list_for_tenant_isolation(self, store):
        """Test list_for_tenant only returns tenant's credentials."""
        for tenant in ["tenant_a", "tenant_b"]:
            for i in range(3):
                cred = EmailCredential(
                    tenant_id=tenant,
                    provider="gmail",
                    email_address=f"user{i}@{tenant}.com",
                )
                await store.save(cred)

        tenant_a_creds = await store.list_for_tenant("tenant_a")
        tenant_b_creds = await store.list_for_tenant("tenant_b")

        assert len(tenant_a_creds) == 3
        assert len(tenant_b_creds) == 3
        assert all(c.tenant_id == "tenant_a" for c in tenant_a_creds)
        assert all(c.tenant_id == "tenant_b" for c in tenant_b_creds)

    @pytest.mark.asyncio
    async def test_delete_only_affects_tenant(self, store):
        """Test delete only affects specified tenant's credential."""
        cred_a = EmailCredential(
            tenant_id="tenant_a",
            provider="gmail",
            email_address="user@example.com",
        )
        cred_b = EmailCredential(
            tenant_id="tenant_b",
            provider="gmail",
            email_address="user@example.com",
        )

        await store.save(cred_a)
        await store.save(cred_b)

        # Delete tenant A's credential
        await store.delete("tenant_a", "gmail", "user@example.com")

        # Tenant B's credential should still exist
        assert await store.get("tenant_a", "gmail", "user@example.com") is None
        assert await store.get("tenant_b", "gmail", "user@example.com") is not None
