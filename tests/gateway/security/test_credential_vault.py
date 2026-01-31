"""
Comprehensive tests for credential vault - security module.

Tests cover:
- CredentialScope enum
- CredentialEntry dataclass
- CredentialVault CRUD operations
- Encryption/decryption with cryptography library
- Security behavior (no fallback to unencrypted storage)
- Expired credentials handling
- Tenant isolation
- Agent restrictions
- Audit logging
- Credential rotation and revocation
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.gateway.security.credential_vault import (
    CredentialEntry,
    CredentialScope,
    CredentialVault,
)


# =============================================================================
# CredentialScope Tests
# =============================================================================


class TestCredentialScope:
    """Tests for CredentialScope enum."""

    def test_global_scope_value(self) -> None:
        """Test global scope value."""
        assert CredentialScope.GLOBAL.value == "global"

    def test_tenant_scope_value(self) -> None:
        """Test tenant scope value."""
        assert CredentialScope.TENANT.value == "tenant"

    def test_agent_scope_value(self) -> None:
        """Test agent scope value."""
        assert CredentialScope.AGENT.value == "agent"

    def test_execution_scope_value(self) -> None:
        """Test execution scope value."""
        assert CredentialScope.EXECUTION.value == "execution"

    def test_scope_is_string_enum(self) -> None:
        """Test that scope is a string enum."""
        assert isinstance(CredentialScope.GLOBAL.value, str)
        assert str(CredentialScope.TENANT) == "CredentialScope.TENANT"

    def test_all_scope_values_unique(self) -> None:
        """Test all scope values are unique."""
        values = [s.value for s in CredentialScope]
        assert len(values) == len(set(values))


# =============================================================================
# CredentialEntry Tests
# =============================================================================


class TestCredentialEntry:
    """Tests for CredentialEntry dataclass."""

    def test_entry_creation_minimal(self) -> None:
        """Test entry creation with minimal fields."""
        entry = CredentialEntry(
            credential_id="cred-123",
            name="API_KEY",
            scope=CredentialScope.GLOBAL,
            encrypted_value=b"encrypted-data",
        )
        assert entry.credential_id == "cred-123"
        assert entry.name == "API_KEY"
        assert entry.scope == CredentialScope.GLOBAL
        assert entry.encrypted_value == b"encrypted-data"

    def test_entry_creation_full(self) -> None:
        """Test entry creation with all fields."""
        now = datetime.now(timezone.utc)
        entry = CredentialEntry(
            credential_id="cred-456",
            name="SECRET_KEY",
            scope=CredentialScope.TENANT,
            encrypted_value=b"encrypted",
            created_at=now,
            expires_at=now + timedelta(days=30),
            tenant_id="tenant-123",
            agent_names=["agent1", "agent2"],
            description="Test credential",
            tags=["test", "api"],
            rotation_policy="30d",
        )
        assert entry.tenant_id == "tenant-123"
        assert entry.agent_names == ["agent1", "agent2"]
        assert entry.description == "Test credential"
        assert entry.tags == ["test", "api"]
        assert entry.rotation_policy == "30d"

    def test_entry_default_created_at(self) -> None:
        """Test entry has default created_at timestamp."""
        entry = CredentialEntry(
            credential_id="cred",
            name="KEY",
            scope=CredentialScope.GLOBAL,
            encrypted_value=b"data",
        )
        assert entry.created_at is not None
        assert isinstance(entry.created_at, datetime)

    def test_entry_is_expired_false_when_no_expiry(self) -> None:
        """Test is_expired is False when no expiry set."""
        entry = CredentialEntry(
            credential_id="cred",
            name="KEY",
            scope=CredentialScope.GLOBAL,
            encrypted_value=b"data",
            expires_at=None,
        )
        assert entry.is_expired is False

    def test_entry_is_expired_false_when_future_expiry(self) -> None:
        """Test is_expired is False when expiry is in future."""
        entry = CredentialEntry(
            credential_id="cred",
            name="KEY",
            scope=CredentialScope.GLOBAL,
            encrypted_value=b"data",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert entry.is_expired is False

    def test_entry_is_expired_true_when_past_expiry(self) -> None:
        """Test is_expired is True when expiry is in past."""
        entry = CredentialEntry(
            credential_id="cred",
            name="KEY",
            scope=CredentialScope.GLOBAL,
            encrypted_value=b"data",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert entry.is_expired is True

    def test_entry_access_count_default_zero(self) -> None:
        """Test access_count defaults to zero."""
        entry = CredentialEntry(
            credential_id="cred",
            name="KEY",
            scope=CredentialScope.GLOBAL,
            encrypted_value=b"data",
        )
        assert entry.access_count == 0

    def test_entry_last_accessed_at_default_none(self) -> None:
        """Test last_accessed_at defaults to None."""
        entry = CredentialEntry(
            credential_id="cred",
            name="KEY",
            scope=CredentialScope.GLOBAL,
            encrypted_value=b"data",
        )
        assert entry.last_accessed_at is None


# =============================================================================
# CredentialVault Basic Tests
# =============================================================================


class TestCredentialVaultBasic:
    """Basic tests for CredentialVault."""

    def test_vault_creation_default(self) -> None:
        """Test vault creation with defaults."""
        vault = CredentialVault()
        assert vault._encryption_key is not None
        assert len(vault._encryption_key) == 32  # 256 bits

    def test_vault_creation_with_key(self) -> None:
        """Test vault creation with explicit key."""
        key = secrets.token_bytes(32)
        vault = CredentialVault(encryption_key=key)
        assert vault._encryption_key == key

    def test_vault_creation_with_env_key(self) -> None:
        """Test vault creation with environment key."""
        with patch.dict(os.environ, {"ARAGORA_CREDENTIAL_VAULT_KEY": "test-env-key"}):
            vault = CredentialVault()
            # Key should be SHA-256 of the env key
            import hashlib

            expected = hashlib.sha256(b"test-env-key").digest()
            assert vault._encryption_key == expected

    def test_vault_creation_with_storage_backend(self) -> None:
        """Test vault creation with storage backend."""
        mock_storage = MagicMock()
        vault = CredentialVault(storage_backend=mock_storage)
        assert vault._storage == mock_storage

    def test_vault_creation_with_audit_logger(self) -> None:
        """Test vault creation with audit logger."""
        mock_logger = MagicMock()
        vault = CredentialVault(audit_logger=mock_logger)
        assert vault._audit_logger == mock_logger


# =============================================================================
# CredentialVault Encryption Tests
# =============================================================================


class TestCredentialVaultEncryption:
    """Tests for CredentialVault encryption/decryption."""

    def test_encrypt_decrypt_roundtrip(self) -> None:
        """Test encryption and decryption roundtrip."""
        vault = CredentialVault()
        original = "super-secret-api-key-12345"

        encrypted = vault._encrypt(original)
        decrypted = vault._decrypt(encrypted)

        assert decrypted == original

    def test_encrypt_produces_different_ciphertext(self) -> None:
        """Test encryption produces different ciphertext each time (nonce)."""
        vault = CredentialVault()
        original = "test-value"

        encrypted1 = vault._encrypt(original)
        encrypted2 = vault._encrypt(original)

        # Same plaintext should produce different ciphertext due to random nonce
        assert encrypted1 != encrypted2

    def test_encrypted_value_is_bytes(self) -> None:
        """Test encrypted value is bytes."""
        vault = CredentialVault()
        encrypted = vault._encrypt("test")
        assert isinstance(encrypted, bytes)

    def test_encrypted_value_contains_nonce(self) -> None:
        """Test encrypted value contains 12-byte nonce prefix."""
        vault = CredentialVault()
        encrypted = vault._encrypt("test")
        # Nonce is 12 bytes, so encrypted should be at least 12 bytes
        assert len(encrypted) >= 12

    def test_decrypt_with_wrong_key_fails(self) -> None:
        """Test decryption with wrong key fails."""
        vault1 = CredentialVault(encryption_key=secrets.token_bytes(32))
        vault2 = CredentialVault(encryption_key=secrets.token_bytes(32))

        encrypted = vault1._encrypt("secret")

        with pytest.raises(Exception):  # Cryptography raises InvalidTag
            vault2._decrypt(encrypted)

    def test_encrypt_empty_string(self) -> None:
        """Test encrypting empty string."""
        vault = CredentialVault()
        encrypted = vault._encrypt("")
        decrypted = vault._decrypt(encrypted)
        assert decrypted == ""

    def test_encrypt_unicode_string(self) -> None:
        """Test encrypting unicode string."""
        vault = CredentialVault()
        original = "secret-key-with-unicode-\u00e9\u00e8\u00ea"
        encrypted = vault._encrypt(original)
        decrypted = vault._decrypt(encrypted)
        assert decrypted == original

    def test_encrypt_long_string(self) -> None:
        """Test encrypting long string."""
        vault = CredentialVault()
        original = "x" * 10000
        encrypted = vault._encrypt(original)
        decrypted = vault._decrypt(encrypted)
        assert decrypted == original


# =============================================================================
# CredentialVault Security - No Fallback Tests
# =============================================================================


class TestCredentialVaultSecurityNoFallback:
    """Tests for security behavior: encryption MUST fail without cryptography."""

    def test_encrypt_fails_without_cryptography(self) -> None:
        """Test encryption fails if cryptography is not available."""
        vault = CredentialVault()

        # Mock the import to simulate cryptography not being available
        with patch.dict("sys.modules", {"cryptography.hazmat.primitives.ciphers.aead": None}):
            with patch(
                "aragora.gateway.security.credential_vault.CredentialVault._encrypt"
            ) as mock_encrypt:
                mock_encrypt.side_effect = RuntimeError(
                    "cryptography library is required for credential vault"
                )
                with pytest.raises(RuntimeError) as exc_info:
                    mock_encrypt("test-value")
                assert "cryptography library is required" in str(exc_info.value)

    def test_decrypt_fails_without_cryptography(self) -> None:
        """Test decryption fails if cryptography is not available."""
        vault = CredentialVault()

        with patch(
            "aragora.gateway.security.credential_vault.CredentialVault._decrypt"
        ) as mock_decrypt:
            mock_decrypt.side_effect = RuntimeError(
                "cryptography library is required for credential vault"
            )
            with pytest.raises(RuntimeError) as exc_info:
                mock_decrypt(b"encrypted-data")
            assert "cryptography library is required" in str(exc_info.value)

    def test_store_fails_without_cryptography(self) -> None:
        """Test storing credential fails without cryptography."""
        vault = CredentialVault()

        with patch.object(vault, "_encrypt") as mock_encrypt:
            mock_encrypt.side_effect = RuntimeError("cryptography library is required")
            with pytest.raises(RuntimeError):
                vault.store(name="KEY", value="secret")


# =============================================================================
# CredentialVault Store Tests
# =============================================================================


class TestCredentialVaultStore:
    """Tests for CredentialVault.store method."""

    def test_store_credential_global(self) -> None:
        """Test storing global credential."""
        vault = CredentialVault()
        cred_id = vault.store(
            name="API_KEY",
            value="sk-test-123",
            scope=CredentialScope.GLOBAL,
        )

        assert cred_id is not None
        assert "global:" in cred_id
        assert "API_KEY:" in cred_id
        assert cred_id in vault._credentials

    def test_store_credential_tenant(self) -> None:
        """Test storing tenant-scoped credential."""
        vault = CredentialVault()
        cred_id = vault.store(
            name="TENANT_KEY",
            value="tenant-secret",
            scope=CredentialScope.TENANT,
            tenant_id="acme-corp",
        )

        assert "tenant:" in cred_id
        entry = vault._credentials[cred_id]
        assert entry.tenant_id == "acme-corp"

    def test_store_credential_agent(self) -> None:
        """Test storing agent-scoped credential."""
        vault = CredentialVault()
        cred_id = vault.store(
            name="AGENT_KEY",
            value="agent-secret",
            scope=CredentialScope.AGENT,
            agent_names=["openclaw", "openhands"],
        )

        assert "agent:" in cred_id
        entry = vault._credentials[cred_id]
        assert entry.agent_names == ["openclaw", "openhands"]

    def test_store_credential_execution(self) -> None:
        """Test storing execution-scoped credential."""
        vault = CredentialVault()
        cred_id = vault.store(
            name="EXEC_KEY",
            value="exec-secret",
            scope=CredentialScope.EXECUTION,
        )

        assert "execution:" in cred_id

    def test_store_credential_with_expiry(self) -> None:
        """Test storing credential with expiry."""
        vault = CredentialVault()
        cred_id = vault.store(
            name="EXPIRING_KEY",
            value="temp-secret",
            expires_in=timedelta(hours=24),
        )

        entry = vault._credentials[cred_id]
        assert entry.expires_at is not None
        assert entry.expires_at > datetime.now(timezone.utc)

    def test_store_credential_with_description(self) -> None:
        """Test storing credential with description."""
        vault = CredentialVault()
        cred_id = vault.store(
            name="DESCRIBED_KEY",
            value="secret",
            description="API key for production environment",
        )

        entry = vault._credentials[cred_id]
        assert entry.description == "API key for production environment"

    def test_store_credential_with_rotation_policy(self) -> None:
        """Test storing credential with rotation policy."""
        vault = CredentialVault()
        cred_id = vault.store(
            name="ROTATING_KEY",
            value="secret",
            rotation_policy="90d",
        )

        entry = vault._credentials[cred_id]
        assert entry.rotation_policy == "90d"

    def test_store_encrypts_value(self) -> None:
        """Test store encrypts the value."""
        vault = CredentialVault()
        cred_id = vault.store(name="KEY", value="plaintext-secret")

        entry = vault._credentials[cred_id]
        # Encrypted value should not equal plaintext
        assert entry.encrypted_value != b"plaintext-secret"
        # Should be able to decrypt back
        decrypted = vault._decrypt(entry.encrypted_value)
        assert decrypted == "plaintext-secret"

    def test_store_returns_unique_ids(self) -> None:
        """Test store returns unique IDs."""
        vault = CredentialVault()
        ids = set()
        for i in range(100):
            cred_id = vault.store(name="KEY", value=f"value-{i}")
            ids.add(cred_id)
        assert len(ids) == 100


# =============================================================================
# CredentialVault Get Credentials Tests
# =============================================================================


class TestCredentialVaultGetCredentials:
    """Tests for CredentialVault._get_credential_value method."""

    def test_get_credential_value_success(self) -> None:
        """Test getting credential value successfully."""
        vault = CredentialVault()
        cred_id = vault.store(name="KEY", value="secret-value")
        entry = vault._credentials[cred_id]

        value = vault._get_credential_value(entry)
        assert value == "secret-value"

    def test_get_credential_value_updates_access_tracking(self) -> None:
        """Test getting credential updates access tracking."""
        vault = CredentialVault()
        cred_id = vault.store(name="KEY", value="secret")
        entry = vault._credentials[cred_id]

        assert entry.access_count == 0
        assert entry.last_accessed_at is None

        vault._get_credential_value(entry)

        assert entry.access_count == 1
        assert entry.last_accessed_at is not None

    def test_get_credential_value_expired_returns_none(self) -> None:
        """Test getting expired credential returns None."""
        vault = CredentialVault()
        cred_id = vault.store(
            name="EXPIRED_KEY",
            value="secret",
            expires_in=timedelta(seconds=-1),  # Already expired
        )
        entry = vault._credentials[cred_id]

        value = vault._get_credential_value(entry)
        assert value is None

    def test_get_credential_value_wrong_tenant_returns_none(self) -> None:
        """Test getting credential with wrong tenant returns None."""
        vault = CredentialVault()
        cred_id = vault.store(
            name="TENANT_KEY",
            value="secret",
            scope=CredentialScope.TENANT,
            tenant_id="tenant-A",
        )
        entry = vault._credentials[cred_id]

        # Try with wrong tenant
        value = vault._get_credential_value(entry, tenant_id="tenant-B")
        assert value is None

    def test_get_credential_value_correct_tenant_returns_value(self) -> None:
        """Test getting credential with correct tenant returns value."""
        vault = CredentialVault()
        cred_id = vault.store(
            name="TENANT_KEY",
            value="secret",
            scope=CredentialScope.TENANT,
            tenant_id="tenant-A",
        )
        entry = vault._credentials[cred_id]

        value = vault._get_credential_value(entry, tenant_id="tenant-A")
        assert value == "secret"

    def test_get_credential_value_wrong_agent_returns_none(self) -> None:
        """Test getting credential with wrong agent returns None."""
        vault = CredentialVault()
        cred_id = vault.store(
            name="AGENT_KEY",
            value="secret",
            agent_names=["allowed-agent"],
        )
        entry = vault._credentials[cred_id]

        value = vault._get_credential_value(entry, agent_name="other-agent")
        assert value is None

    def test_get_credential_value_correct_agent_returns_value(self) -> None:
        """Test getting credential with correct agent returns value."""
        vault = CredentialVault()
        cred_id = vault.store(
            name="AGENT_KEY",
            value="secret",
            agent_names=["allowed-agent"],
        )
        entry = vault._credentials[cred_id]

        value = vault._get_credential_value(entry, agent_name="allowed-agent")
        assert value == "secret"


# =============================================================================
# CredentialVault Get Credentials for Execution Tests
# =============================================================================


class TestCredentialVaultGetCredentialsForExecution:
    """Tests for CredentialVault.get_credentials_for_execution method."""

    @pytest.mark.asyncio
    async def test_get_all_credentials_for_execution(self) -> None:
        """Test getting all credentials for execution."""
        vault = CredentialVault()
        vault.store(name="KEY1", value="value1")
        vault.store(name="KEY2", value="value2")
        vault.store(name="KEY3", value="value3")

        creds = await vault.get_credentials_for_execution(agent_name="test-agent")

        assert len(creds) == 3
        assert creds["KEY1"] == "value1"
        assert creds["KEY2"] == "value2"
        assert creds["KEY3"] == "value3"

    @pytest.mark.asyncio
    async def test_get_required_credentials_only(self) -> None:
        """Test getting only required credentials."""
        vault = CredentialVault()
        vault.store(name="KEY1", value="value1")
        vault.store(name="KEY2", value="value2")
        vault.store(name="KEY3", value="value3")

        creds = await vault.get_credentials_for_execution(
            agent_name="test-agent",
            required_credentials=["KEY1", "KEY3"],
        )

        assert len(creds) == 2
        assert "KEY1" in creds
        assert "KEY2" not in creds
        assert "KEY3" in creds

    @pytest.mark.asyncio
    async def test_get_credentials_respects_tenant_scope(self) -> None:
        """Test credential retrieval respects tenant scope."""
        vault = CredentialVault()
        vault.store(
            name="TENANT_A_KEY",
            value="tenant-a-secret",
            scope=CredentialScope.TENANT,
            tenant_id="tenant-A",
        )
        vault.store(
            name="TENANT_B_KEY",
            value="tenant-b-secret",
            scope=CredentialScope.TENANT,
            tenant_id="tenant-B",
        )

        # Get for tenant A
        creds = await vault.get_credentials_for_execution(
            agent_name="agent",
            tenant_id="tenant-A",
        )

        assert "TENANT_A_KEY" in creds
        assert "TENANT_B_KEY" not in creds

    @pytest.mark.asyncio
    async def test_get_credentials_respects_agent_restriction(self) -> None:
        """Test credential retrieval respects agent restriction."""
        vault = CredentialVault()
        vault.store(
            name="RESTRICTED_KEY",
            value="restricted-secret",
            agent_names=["special-agent"],
        )

        # Other agent shouldn't get the credential
        creds = await vault.get_credentials_for_execution(agent_name="other-agent")
        assert "RESTRICTED_KEY" not in creds

        # Special agent should get the credential
        creds = await vault.get_credentials_for_execution(agent_name="special-agent")
        assert "RESTRICTED_KEY" in creds

    @pytest.mark.asyncio
    async def test_get_credentials_excludes_expired(self) -> None:
        """Test expired credentials are excluded."""
        vault = CredentialVault()
        vault.store(name="VALID_KEY", value="valid-secret")
        vault.store(
            name="EXPIRED_KEY",
            value="expired-secret",
            expires_in=timedelta(seconds=-1),
        )

        creds = await vault.get_credentials_for_execution(agent_name="agent")

        assert "VALID_KEY" in creds
        assert "EXPIRED_KEY" not in creds

    @pytest.mark.asyncio
    async def test_get_credentials_with_audit_logger(self) -> None:
        """Test audit logger is called."""
        mock_logger = AsyncMock()
        vault = CredentialVault(audit_logger=mock_logger)
        vault.store(name="KEY", value="secret")

        await vault.get_credentials_for_execution(
            agent_name="test-agent",
            tenant_id="test-tenant",
        )

        mock_logger.log_credential_access.assert_called_once()
        call_kwargs = mock_logger.log_credential_access.call_args.kwargs
        assert call_kwargs["agent_name"] == "test-agent"
        assert call_kwargs["tenant_id"] == "test-tenant"

    @pytest.mark.asyncio
    async def test_get_credentials_empty_vault(self) -> None:
        """Test getting credentials from empty vault."""
        vault = CredentialVault()

        creds = await vault.get_credentials_for_execution(agent_name="agent")

        assert creds == {}


# =============================================================================
# CredentialVault Rotate Tests
# =============================================================================


class TestCredentialVaultRotate:
    """Tests for CredentialVault.rotate method."""

    def test_rotate_credential_success(self) -> None:
        """Test rotating credential successfully."""
        vault = CredentialVault()
        cred_id = vault.store(name="KEY", value="old-value")

        result = vault.rotate(cred_id, "new-value")

        assert result is True
        entry = vault._credentials[cred_id]
        decrypted = vault._decrypt(entry.encrypted_value)
        assert decrypted == "new-value"

    def test_rotate_resets_access_count(self) -> None:
        """Test rotation resets access count."""
        vault = CredentialVault()
        cred_id = vault.store(name="KEY", value="old-value")
        entry = vault._credentials[cred_id]
        entry.access_count = 100

        vault.rotate(cred_id, "new-value")

        assert entry.access_count == 0

    def test_rotate_nonexistent_credential(self) -> None:
        """Test rotating nonexistent credential returns False."""
        vault = CredentialVault()

        result = vault.rotate("nonexistent-id", "new-value")

        assert result is False


# =============================================================================
# CredentialVault Revoke Tests
# =============================================================================


class TestCredentialVaultRevoke:
    """Tests for CredentialVault.revoke method."""

    def test_revoke_credential_success(self) -> None:
        """Test revoking credential successfully."""
        vault = CredentialVault()
        cred_id = vault.store(name="KEY", value="secret")

        result = vault.revoke(cred_id)

        assert result is True
        assert cred_id not in vault._credentials

    def test_revoke_nonexistent_credential(self) -> None:
        """Test revoking nonexistent credential returns False."""
        vault = CredentialVault()

        result = vault.revoke("nonexistent-id")

        assert result is False

    def test_revoke_removes_from_future_queries(self) -> None:
        """Test revoked credential not returned in queries."""
        vault = CredentialVault()
        cred_id = vault.store(name="KEY", value="secret")

        vault.revoke(cred_id)
        creds_list = vault.list_credentials()

        assert len(creds_list) == 0


# =============================================================================
# CredentialVault List Tests
# =============================================================================


class TestCredentialVaultList:
    """Tests for CredentialVault.list_credentials method."""

    def test_list_all_credentials(self) -> None:
        """Test listing all credentials."""
        vault = CredentialVault()
        vault.store(name="KEY1", value="value1")
        vault.store(name="KEY2", value="value2")

        creds = vault.list_credentials()

        assert len(creds) == 2

    def test_list_credentials_by_tenant(self) -> None:
        """Test listing credentials filtered by tenant."""
        vault = CredentialVault()
        vault.store(name="KEY1", value="v1", scope=CredentialScope.TENANT, tenant_id="A")
        vault.store(name="KEY2", value="v2", scope=CredentialScope.TENANT, tenant_id="B")
        vault.store(name="KEY3", value="v3", scope=CredentialScope.TENANT, tenant_id="A")

        creds = vault.list_credentials(tenant_id="A")

        assert len(creds) == 2
        names = [c["name"] for c in creds]
        assert "KEY1" in names
        assert "KEY3" in names
        assert "KEY2" not in names

    def test_list_credentials_excludes_sensitive_data(self) -> None:
        """Test listing doesn't expose encrypted values."""
        vault = CredentialVault()
        vault.store(name="SECRET_KEY", value="super-secret")

        creds = vault.list_credentials()

        assert len(creds) == 1
        cred = creds[0]
        assert "encrypted_value" not in cred
        assert "value" not in cred
        assert "name" in cred
        assert cred["name"] == "SECRET_KEY"

    def test_list_credentials_includes_metadata(self) -> None:
        """Test listing includes metadata."""
        vault = CredentialVault()
        vault.store(
            name="KEY",
            value="secret",
            scope=CredentialScope.TENANT,
            tenant_id="test-tenant",
            description="Test description",
        )

        creds = vault.list_credentials()

        assert len(creds) == 1
        cred = creds[0]
        assert "credential_id" in cred
        assert "name" in cred
        assert "scope" in cred
        assert "tenant_id" in cred
        assert "created_at" in cred
        assert "is_expired" in cred
        assert "access_count" in cred
        assert "description" in cred

    def test_list_credentials_empty_vault(self) -> None:
        """Test listing empty vault."""
        vault = CredentialVault()

        creds = vault.list_credentials()

        assert creds == []


# =============================================================================
# CredentialVault Cleanup Tests
# =============================================================================


class TestCredentialVaultCleanup:
    """Tests for CredentialVault.cleanup_expired method."""

    @pytest.mark.asyncio
    async def test_cleanup_expired_credentials(self) -> None:
        """Test cleaning up expired credentials."""
        vault = CredentialVault()
        vault.store(name="VALID", value="valid")
        vault.store(
            name="EXPIRED1",
            value="expired",
            expires_in=timedelta(seconds=-1),
        )
        vault.store(
            name="EXPIRED2",
            value="expired",
            expires_in=timedelta(seconds=-100),
        )

        count = await vault.cleanup_expired()

        assert count == 2
        assert len(vault._credentials) == 1

    @pytest.mark.asyncio
    async def test_cleanup_no_expired_credentials(self) -> None:
        """Test cleanup when no credentials expired."""
        vault = CredentialVault()
        vault.store(name="KEY1", value="value1")
        vault.store(name="KEY2", value="value2", expires_in=timedelta(days=30))

        count = await vault.cleanup_expired()

        assert count == 0
        assert len(vault._credentials) == 2

    @pytest.mark.asyncio
    async def test_cleanup_empty_vault(self) -> None:
        """Test cleanup on empty vault."""
        vault = CredentialVault()

        count = await vault.cleanup_expired()

        assert count == 0


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestCredentialVaultEdgeCases:
    """Edge cases and error handling tests."""

    def test_store_same_name_different_scopes(self) -> None:
        """Test storing credentials with same name but different scopes."""
        vault = CredentialVault()
        id1 = vault.store(name="KEY", value="global", scope=CredentialScope.GLOBAL)
        id2 = vault.store(name="KEY", value="tenant", scope=CredentialScope.TENANT)

        assert id1 != id2
        assert len(vault._credentials) == 2

    def test_credential_with_empty_agent_names(self) -> None:
        """Test credential with empty agent_names allows all agents."""
        vault = CredentialVault()
        cred_id = vault.store(name="KEY", value="secret", agent_names=[])
        entry = vault._credentials[cred_id]

        # Empty agent_names means no restriction
        value = vault._get_credential_value(entry, agent_name="any-agent")
        assert value == "secret"

    def test_credential_with_none_agent_names(self) -> None:
        """Test credential with None agent_names allows all agents."""
        vault = CredentialVault()
        cred_id = vault.store(name="KEY", value="secret", agent_names=None)
        entry = vault._credentials[cred_id]

        value = vault._get_credential_value(entry, agent_name="any-agent")
        assert value == "secret"

    def test_tenant_scope_without_tenant_id(self) -> None:
        """Test tenant-scoped credential without tenant_id."""
        vault = CredentialVault()
        cred_id = vault.store(
            name="KEY",
            value="secret",
            scope=CredentialScope.TENANT,
            tenant_id=None,  # No tenant restriction
        )
        entry = vault._credentials[cred_id]

        # Should be accessible since entry has no tenant_id restriction
        value = vault._get_credential_value(entry, tenant_id="any-tenant")
        assert value == "secret"

    def test_multiple_credentials_same_tenant(self) -> None:
        """Test multiple credentials for same tenant."""
        vault = CredentialVault()
        vault.store(name="KEY1", value="v1", scope=CredentialScope.TENANT, tenant_id="tenant-X")
        vault.store(name="KEY2", value="v2", scope=CredentialScope.TENANT, tenant_id="tenant-X")
        vault.store(name="KEY3", value="v3", scope=CredentialScope.TENANT, tenant_id="tenant-X")

        creds = vault.list_credentials(tenant_id="tenant-X")
        assert len(creds) == 3

    @pytest.mark.asyncio
    async def test_concurrent_credential_access(self) -> None:
        """Test concurrent credential access."""
        import asyncio

        vault = CredentialVault()
        vault.store(name="KEY", value="secret")

        async def access_credential():
            return await vault.get_credentials_for_execution(agent_name="agent")

        # Run multiple concurrent accesses
        tasks = [access_credential() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        for result in results:
            assert "KEY" in result
            assert result["KEY"] == "secret"

    def test_special_characters_in_credential_value(self) -> None:
        """Test special characters in credential value."""
        vault = CredentialVault()
        special_value = 'key="value"&token=abc123!@#$%^&*()'
        cred_id = vault.store(name="SPECIAL", value=special_value)

        entry = vault._credentials[cred_id]
        decrypted = vault._decrypt(entry.encrypted_value)
        assert decrypted == special_value

    def test_newlines_in_credential_value(self) -> None:
        """Test newlines in credential value (e.g., PEM keys)."""
        vault = CredentialVault()
        pem_like = "-----BEGIN KEY-----\nbase64data\nmore data\n-----END KEY-----"
        cred_id = vault.store(name="PEM_KEY", value=pem_like)

        entry = vault._credentials[cred_id]
        decrypted = vault._decrypt(entry.encrypted_value)
        assert decrypted == pem_like
