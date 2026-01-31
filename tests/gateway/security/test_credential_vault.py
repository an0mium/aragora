"""Tests for credential vault."""

import pytest
from datetime import timedelta

from aragora.gateway.security.credential_vault import (
    CredentialVault,
    CredentialScope,
    CredentialEntry,
)


class TestCredentialScope:
    """Tests for CredentialScope enum."""

    def test_scope_values(self):
        """Test scope values."""
        assert CredentialScope.GLOBAL.value == "global"
        assert CredentialScope.TENANT.value == "tenant"
        assert CredentialScope.AGENT.value == "agent"
        assert CredentialScope.EXECUTION.value == "execution"


class TestCredentialEntry:
    """Tests for CredentialEntry."""

    def test_entry_creation(self):
        """Test entry creation."""
        entry = CredentialEntry(
            credential_id="test-id",
            name="API_KEY",
            scope=CredentialScope.TENANT,
            encrypted_value=b"encrypted",
            tenant_id="tenant-123",
        )
        assert entry.credential_id == "test-id"
        assert entry.name == "API_KEY"
        assert entry.scope == CredentialScope.TENANT
        assert entry.is_expired is False

    def test_entry_not_expired_no_expiry(self):
        """Test entry without expiry is not expired."""
        entry = CredentialEntry(
            credential_id="test-id",
            name="API_KEY",
            scope=CredentialScope.GLOBAL,
            encrypted_value=b"encrypted",
        )
        assert entry.is_expired is False


class TestCredentialVault:
    """Tests for CredentialVault."""

    def test_store_credential(self):
        """Test storing a credential."""
        vault = CredentialVault()
        credential_id = vault.store(
            name="API_KEY",
            value="secret-value",
            scope=CredentialScope.GLOBAL,
        )

        assert credential_id is not None
        assert "global:" in credential_id
        assert credential_id in vault._credentials

    def test_store_credential_with_tenant(self):
        """Test storing tenant-scoped credential."""
        vault = CredentialVault()
        credential_id = vault.store(
            name="API_KEY",
            value="secret-value",
            scope=CredentialScope.TENANT,
            tenant_id="tenant-123",
        )

        assert "tenant:" in credential_id
        entry = vault._credentials[credential_id]
        assert entry.tenant_id == "tenant-123"

    def test_store_credential_with_agent_restriction(self):
        """Test storing credential with agent restriction."""
        vault = CredentialVault()
        credential_id = vault.store(
            name="API_KEY",
            value="secret-value",
            scope=CredentialScope.AGENT,
            agent_names=["openclaw", "openhands"],
        )

        entry = vault._credentials[credential_id]
        assert entry.agent_names == ["openclaw", "openhands"]

    def test_store_credential_with_expiry(self):
        """Test storing credential with expiry."""
        vault = CredentialVault()
        credential_id = vault.store(
            name="API_KEY",
            value="secret-value",
            expires_in=timedelta(hours=1),
        )

        entry = vault._credentials[credential_id]
        assert entry.expires_at is not None
        assert entry.is_expired is False

    @pytest.mark.asyncio
    async def test_get_credentials_for_execution(self):
        """Test getting credentials for execution."""
        vault = CredentialVault()
        vault.store(
            name="OPENAI_API_KEY",
            value="sk-test-123",
            scope=CredentialScope.GLOBAL,
        )
        vault.store(
            name="OTHER_KEY",
            value="other-value",
            scope=CredentialScope.GLOBAL,
        )

        creds = await vault.get_credentials_for_execution(
            agent_name="test-agent",
            required_credentials=["OPENAI_API_KEY"],
        )

        assert "OPENAI_API_KEY" in creds
        assert creds["OPENAI_API_KEY"] == "sk-test-123"
        assert "OTHER_KEY" not in creds  # Not requested

    @pytest.mark.asyncio
    async def test_get_credentials_all(self):
        """Test getting all applicable credentials."""
        vault = CredentialVault()
        vault.store(name="KEY1", value="value1", scope=CredentialScope.GLOBAL)
        vault.store(name="KEY2", value="value2", scope=CredentialScope.GLOBAL)

        creds = await vault.get_credentials_for_execution(agent_name="test-agent")

        assert "KEY1" in creds
        assert "KEY2" in creds

    @pytest.mark.asyncio
    async def test_get_credentials_respects_tenant_scope(self):
        """Test credential retrieval respects tenant scope."""
        vault = CredentialVault()
        vault.store(
            name="TENANT_KEY",
            value="tenant-secret",
            scope=CredentialScope.TENANT,
            tenant_id="tenant-A",
        )

        # Wrong tenant - shouldn't get credential
        creds = await vault.get_credentials_for_execution(
            agent_name="test-agent",
            tenant_id="tenant-B",
        )
        assert "TENANT_KEY" not in creds

        # Correct tenant - should get credential
        creds = await vault.get_credentials_for_execution(
            agent_name="test-agent",
            tenant_id="tenant-A",
        )
        assert "TENANT_KEY" in creds

    @pytest.mark.asyncio
    async def test_get_credentials_respects_agent_restriction(self):
        """Test credential retrieval respects agent restriction."""
        vault = CredentialVault()
        vault.store(
            name="AGENT_KEY",
            value="agent-secret",
            agent_names=["allowed-agent"],
        )

        # Wrong agent - shouldn't get credential
        creds = await vault.get_credentials_for_execution(
            agent_name="other-agent",
        )
        assert "AGENT_KEY" not in creds

        # Correct agent - should get credential
        creds = await vault.get_credentials_for_execution(
            agent_name="allowed-agent",
        )
        assert "AGENT_KEY" in creds

    def test_rotate_credential(self):
        """Test rotating a credential."""
        vault = CredentialVault()
        credential_id = vault.store(
            name="API_KEY",
            value="old-value",
        )

        result = vault.rotate(credential_id, "new-value")
        assert result is True

        # Verify new value
        entry = vault._credentials[credential_id]
        decrypted = vault._decrypt(entry.encrypted_value)
        assert decrypted == "new-value"

    def test_rotate_nonexistent_credential(self):
        """Test rotating nonexistent credential."""
        vault = CredentialVault()
        result = vault.rotate("nonexistent-id", "new-value")
        assert result is False

    def test_revoke_credential(self):
        """Test revoking a credential."""
        vault = CredentialVault()
        credential_id = vault.store(
            name="API_KEY",
            value="secret",
        )

        result = vault.revoke(credential_id)
        assert result is True
        assert credential_id not in vault._credentials

    def test_revoke_nonexistent_credential(self):
        """Test revoking nonexistent credential."""
        vault = CredentialVault()
        result = vault.revoke("nonexistent-id")
        assert result is False

    def test_list_credentials(self):
        """Test listing credentials."""
        vault = CredentialVault()
        vault.store(name="KEY1", value="value1")
        vault.store(name="KEY2", value="value2", tenant_id="tenant-A")

        # List all
        all_creds = vault.list_credentials()
        assert len(all_creds) == 2

        # List by tenant
        tenant_creds = vault.list_credentials(tenant_id="tenant-A")
        assert len(tenant_creds) == 1
        assert tenant_creds[0]["name"] == "KEY2"

    def test_list_credentials_no_values(self):
        """Test that listing doesn't expose values."""
        vault = CredentialVault()
        vault.store(name="API_KEY", value="super-secret")

        creds = vault.list_credentials()
        assert len(creds) == 1
        assert "value" not in creds[0]
        assert "encrypted_value" not in creds[0]

    @pytest.mark.asyncio
    async def test_cleanup_expired(self):
        """Test cleaning up expired credentials."""
        vault = CredentialVault()
        # Create an expired credential
        credential_id = vault.store(
            name="EXPIRED_KEY",
            value="secret",
            expires_in=timedelta(seconds=-1),  # Already expired
        )

        count = await vault.cleanup_expired()
        assert count == 1
        assert credential_id not in vault._credentials

    def test_encryption_decryption(self):
        """Test encryption and decryption roundtrip."""
        vault = CredentialVault()
        original = "test-secret-value"

        encrypted = vault._encrypt(original)
        decrypted = vault._decrypt(encrypted)

        assert decrypted == original
        assert encrypted != original.encode()  # Should be encrypted
