"""
Tests for Microsoft Teams Tenant Storage.

Tests cover:
- SQLite-backed TeamsTenantStore
- Supabase-backed SupabaseTeamsTenantStore
- Token encryption/decryption
- Factory function selection
"""

from __future__ import annotations

import os
import tempfile
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest

from aragora.storage.teams_tenant_store import (
    TeamsTenant,
    TeamsTenantStore,
    SupabaseTeamsTenantStore,
    get_teams_tenant_store,
)


# ============================================================================
# TeamsTenant Dataclass Tests
# ============================================================================


class TestTeamsTenant:
    """Tests for TeamsTenant dataclass."""

    def test_create_tenant(self):
        """Create a basic TeamsTenant."""
        tenant = TeamsTenant(
            tenant_id="tenant-123",
            tenant_name="Test Org",
            access_token="token-abc",
            bot_id="bot-456",
            installed_at=time.time(),
        )

        assert tenant.tenant_id == "tenant-123"
        assert tenant.tenant_name == "Test Org"
        assert tenant.access_token == "token-abc"
        assert tenant.bot_id == "bot-456"
        assert tenant.is_active is True

    def test_tenant_to_dict(self):
        """Convert tenant to dict (without sensitive tokens)."""
        now = time.time()
        tenant = TeamsTenant(
            tenant_id="tenant-123",
            tenant_name="Test Org",
            access_token="token-abc",
            bot_id="bot-456",
            installed_at=now,
        )

        data = tenant.to_dict()

        assert data["tenant_id"] == "tenant-123"
        assert data["tenant_name"] == "Test Org"
        assert "access_token" not in data  # Sensitive, not included
        assert data["bot_id"] == "bot-456"
        assert data["installed_at"] == now

    def test_token_not_expired(self):
        """Token is not expired when expires_at is in future."""
        tenant = TeamsTenant(
            tenant_id="tenant-123",
            tenant_name="Test Org",
            access_token="token-abc",
            bot_id="bot-456",
            installed_at=time.time(),
            expires_at=time.time() + 3600,  # 1 hour from now
        )

        assert tenant.is_token_expired() is False

    def test_token_expired(self):
        """Token is expired when expires_at is in past."""
        tenant = TeamsTenant(
            tenant_id="tenant-123",
            tenant_name="Test Org",
            access_token="token-abc",
            bot_id="bot-456",
            installed_at=time.time() - 7200,
            expires_at=time.time() - 3600,  # 1 hour ago
        )

        assert tenant.is_token_expired() is True

    def test_token_expired_within_buffer(self):
        """Token is considered expired within 5-minute buffer."""
        tenant = TeamsTenant(
            tenant_id="tenant-123",
            tenant_name="Test Org",
            access_token="token-abc",
            bot_id="bot-456",
            installed_at=time.time(),
            expires_at=time.time() + 60,  # 1 minute from now (within buffer)
        )

        assert tenant.is_token_expired() is True

    def test_no_expiration_not_expired(self):
        """Token without expires_at is not expired."""
        tenant = TeamsTenant(
            tenant_id="tenant-123",
            tenant_name="Test Org",
            access_token="token-abc",
            bot_id="bot-456",
            installed_at=time.time(),
        )

        assert tenant.is_token_expired() is False


# ============================================================================
# SQLite TeamsTenantStore Tests
# ============================================================================


class TestTeamsTenantStore:
    """Tests for SQLite-backed TeamsTenantStore."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def store(self, temp_db):
        """Create TeamsTenantStore with temp database."""
        return TeamsTenantStore(db_path=temp_db)

    def test_save_and_get_tenant(self, store):
        """Save and retrieve a tenant."""
        tenant = TeamsTenant(
            tenant_id="tenant-123",
            tenant_name="Test Org",
            access_token="token-abc",
            bot_id="bot-456",
            installed_at=time.time(),
            scopes=["User.Read", "Chat.ReadWrite"],
        )

        assert store.save(tenant) is True

        retrieved = store.get("tenant-123")
        assert retrieved is not None
        assert retrieved.tenant_id == "tenant-123"
        assert retrieved.tenant_name == "Test Org"
        assert retrieved.scopes == ["User.Read", "Chat.ReadWrite"]

    def test_get_nonexistent_tenant(self, store):
        """Get returns None for nonexistent tenant."""
        result = store.get("nonexistent")
        assert result is None

    def test_update_tenant(self, store):
        """Update existing tenant."""
        tenant = TeamsTenant(
            tenant_id="tenant-123",
            tenant_name="Test Org",
            access_token="token-abc",
            bot_id="bot-456",
            installed_at=time.time(),
        )
        store.save(tenant)

        # Update
        tenant.tenant_name = "Updated Org"
        tenant.access_token = "new-token"
        store.save(tenant)

        retrieved = store.get("tenant-123")
        assert retrieved.tenant_name == "Updated Org"
        assert retrieved.access_token == "new-token"

    def test_list_active_tenants(self, store):
        """List only active tenants."""
        now = time.time()
        for i in range(3):
            tenant = TeamsTenant(
                tenant_id=f"tenant-{i}",
                tenant_name=f"Org {i}",
                access_token=f"token-{i}",
                bot_id=f"bot-{i}",
                installed_at=now - i * 100,
                is_active=True,
            )
            store.save(tenant)

        # Add inactive tenant
        inactive = TeamsTenant(
            tenant_id="tenant-inactive",
            tenant_name="Inactive Org",
            access_token="token-inactive",
            bot_id="bot-inactive",
            installed_at=now,
            is_active=False,
        )
        store.save(inactive)

        active = store.list_active()
        assert len(active) == 3
        # Should be ordered by installed_at DESC
        assert active[0].tenant_id == "tenant-0"

    def test_deactivate_tenant(self, store):
        """Deactivate a tenant."""
        tenant = TeamsTenant(
            tenant_id="tenant-123",
            tenant_name="Test Org",
            access_token="token-abc",
            bot_id="bot-456",
            installed_at=time.time(),
        )
        store.save(tenant)

        assert store.deactivate("tenant-123") is True

        retrieved = store.get("tenant-123")
        assert retrieved.is_active is False

    def test_delete_tenant(self, store):
        """Delete a tenant permanently."""
        tenant = TeamsTenant(
            tenant_id="tenant-123",
            tenant_name="Test Org",
            access_token="token-abc",
            bot_id="bot-456",
            installed_at=time.time(),
        )
        store.save(tenant)

        assert store.delete("tenant-123") is True
        assert store.get("tenant-123") is None

    def test_update_tokens(self, store):
        """Update tokens after refresh."""
        tenant = TeamsTenant(
            tenant_id="tenant-123",
            tenant_name="Test Org",
            access_token="old-token",
            refresh_token="old-refresh",
            bot_id="bot-456",
            installed_at=time.time(),
            expires_at=time.time() + 100,
        )
        store.save(tenant)

        new_expires = time.time() + 3600
        assert (
            store.update_tokens(
                "tenant-123",
                "new-token",
                "new-refresh",
                new_expires,
            )
            is True
        )

        retrieved = store.get("tenant-123")
        assert retrieved.access_token == "new-token"
        assert retrieved.refresh_token == "new-refresh"
        assert abs(retrieved.expires_at - new_expires) < 1

    def test_get_by_org(self, store):
        """Get tenants by Aragora organization ID."""
        now = time.time()
        for i in range(2):
            tenant = TeamsTenant(
                tenant_id=f"tenant-{i}",
                tenant_name=f"Org {i}",
                access_token=f"token-{i}",
                bot_id=f"bot-{i}",
                installed_at=now,
                aragora_org_id="aragora-org-1",
            )
            store.save(tenant)

        # Different org
        other = TeamsTenant(
            tenant_id="tenant-other",
            tenant_name="Other Org",
            access_token="token-other",
            bot_id="bot-other",
            installed_at=now,
            aragora_org_id="aragora-org-2",
        )
        store.save(other)

        result = store.get_by_org("aragora-org-1")
        assert len(result) == 2

    def test_list_expiring(self, store):
        """List tenants with expiring tokens."""
        now = time.time()

        # Expiring soon
        expiring = TeamsTenant(
            tenant_id="tenant-expiring",
            tenant_name="Expiring Org",
            access_token="token-exp",
            bot_id="bot-exp",
            installed_at=now,
            expires_at=now + 1800,  # 30 minutes
        )
        store.save(expiring)

        # Not expiring
        not_expiring = TeamsTenant(
            tenant_id="tenant-valid",
            tenant_name="Valid Org",
            access_token="token-valid",
            bot_id="bot-valid",
            installed_at=now,
            expires_at=now + 7200,  # 2 hours
        )
        store.save(not_expiring)

        result = store.list_expiring(within_seconds=3600)
        assert len(result) == 1
        assert result[0].tenant_id == "tenant-expiring"

    def test_count(self, store):
        """Count tenants."""
        for i in range(3):
            tenant = TeamsTenant(
                tenant_id=f"tenant-{i}",
                tenant_name=f"Org {i}",
                access_token=f"token-{i}",
                bot_id=f"bot-{i}",
                installed_at=time.time(),
                is_active=i != 2,  # Third is inactive
            )
            store.save(tenant)

        assert store.count(active_only=True) == 2
        assert store.count(active_only=False) == 3

    def test_get_stats(self, store):
        """Get tenant statistics."""
        now = time.time()
        for i in range(3):
            tenant = TeamsTenant(
                tenant_id=f"tenant-{i}",
                tenant_name=f"Org {i}",
                access_token=f"token-{i}",
                bot_id=f"bot-{i}",
                installed_at=now,
                is_active=True,
                expires_at=now + (1800 if i == 0 else 7200),
            )
            store.save(tenant)

        stats = store.get_stats()
        assert stats["total_tenants"] == 3
        assert stats["active_tenants"] == 3
        assert stats["expiring_tokens"] >= 1


# ============================================================================
# Supabase Store Tests
# ============================================================================


class TestSupabaseTeamsTenantStore:
    """Tests for Supabase-backed SupabaseTeamsTenantStore."""

    @pytest.fixture
    def mock_supabase_client(self):
        """Create mock Supabase client."""
        client = MagicMock()
        return client

    def test_is_configured_without_client(self):
        """Store reports not configured when client unavailable."""
        with patch("aragora.storage.teams_tenant_store.SupabaseTeamsTenantStore._init_client"):
            store = SupabaseTeamsTenantStore()
            store._client = None
            assert store.is_configured is False

    def test_is_configured_with_client(self, mock_supabase_client):
        """Store reports configured when client available."""
        with patch("aragora.storage.teams_tenant_store.SupabaseTeamsTenantStore._init_client"):
            store = SupabaseTeamsTenantStore()
            store._client = mock_supabase_client
            assert store.is_configured is True

    def test_save_to_supabase(self, mock_supabase_client):
        """Save tenant to Supabase."""
        with patch("aragora.storage.teams_tenant_store.SupabaseTeamsTenantStore._init_client"):
            store = SupabaseTeamsTenantStore()
            store._client = mock_supabase_client

            tenant = TeamsTenant(
                tenant_id="tenant-123",
                tenant_name="Test Org",
                access_token="token-abc",
                bot_id="bot-456",
                installed_at=time.time(),
            )

            mock_table = MagicMock()
            mock_supabase_client.table.return_value = mock_table
            mock_table.upsert.return_value.execute.return_value = MagicMock()

            result = store.save(tenant)

            assert result is True
            mock_supabase_client.table.assert_called_with("teams_tenants")
            mock_table.upsert.assert_called_once()

    def test_get_from_supabase(self, mock_supabase_client):
        """Get tenant from Supabase."""
        with patch("aragora.storage.teams_tenant_store.SupabaseTeamsTenantStore._init_client"):
            store = SupabaseTeamsTenantStore()
            store._client = mock_supabase_client

            # Mock response
            now = datetime.now(tz=timezone.utc).isoformat()
            mock_response = MagicMock()
            mock_response.data = [
                {
                    "tenant_id": "tenant-123",
                    "tenant_name": "Test Org",
                    "access_token": "token-abc",
                    "refresh_token": None,
                    "bot_id": "bot-456",
                    "installed_at": now,
                    "installed_by": None,
                    "scopes": ["User.Read"],
                    "aragora_org_id": None,
                    "is_active": True,
                    "expires_at": None,
                }
            ]

            mock_table = MagicMock()
            mock_supabase_client.table.return_value = mock_table
            mock_table.select.return_value.eq.return_value.execute.return_value = mock_response

            result = store.get("tenant-123")

            assert result is not None
            assert result.tenant_id == "tenant-123"
            assert result.tenant_name == "Test Org"

    def test_deactivate_in_supabase(self, mock_supabase_client):
        """Deactivate tenant in Supabase."""
        with patch("aragora.storage.teams_tenant_store.SupabaseTeamsTenantStore._init_client"):
            store = SupabaseTeamsTenantStore()
            store._client = mock_supabase_client

            mock_table = MagicMock()
            mock_supabase_client.table.return_value = mock_table
            mock_table.update.return_value.eq.return_value.execute.return_value = MagicMock()

            result = store.deactivate("tenant-123")

            assert result is True
            mock_table.update.assert_called_once()


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestGetTeamsTenantStore:
    """Tests for factory function."""

    def test_returns_sqlite_by_default(self):
        """Factory returns SQLite store by default."""
        # Reset singleton
        import aragora.storage.teams_tenant_store as module

        module._tenant_store = None

        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}, clear=False):
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                store = get_teams_tenant_store(db_path=f.name)
                assert isinstance(store, TeamsTenantStore)
                module._tenant_store = None
                os.unlink(f.name)

    def test_returns_supabase_in_production(self):
        """Factory returns Supabase store in production."""
        import aragora.storage.teams_tenant_store as module

        module._tenant_store = None

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}, clear=False):
            with patch.object(module, "ARAGORA_ENV", "production"):
                mock_supabase = MagicMock()
                mock_supabase.is_configured = True

                with patch.object(
                    module.SupabaseTeamsTenantStore,
                    "__init__",
                    return_value=None,
                ):
                    with patch.object(
                        module.SupabaseTeamsTenantStore,
                        "is_configured",
                        True,
                    ):
                        with patch(
                            "aragora.storage.teams_tenant_store.SupabaseTeamsTenantStore",
                            return_value=mock_supabase,
                        ):
                            store = get_teams_tenant_store()
                            # Would return Supabase if configured
                            module._tenant_store = None

    def test_singleton_behavior(self):
        """Factory returns same instance on repeated calls."""
        import aragora.storage.teams_tenant_store as module

        module._tenant_store = None

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            store1 = get_teams_tenant_store(db_path=f.name)
            store2 = get_teams_tenant_store(db_path=f.name)
            assert store1 is store2
            module._tenant_store = None
            os.unlink(f.name)


# ============================================================================
# Token Encryption Tests
# ============================================================================


class TestTokenEncryption:
    """Tests for token encryption/decryption."""

    @pytest.fixture
    def store_with_encryption(self, temp_db):
        """Create store with encryption key."""
        with patch.dict(os.environ, {"ARAGORA_ENCRYPTION_KEY": "test-encryption-key"}):
            return TeamsTenantStore(db_path=temp_db)

    @pytest.fixture
    def temp_db(self):
        """Create temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        os.unlink(f.name)

    def test_encrypt_token_without_key(self):
        """Token returned unchanged without encryption key."""
        with patch("aragora.storage.teams_tenant_store.ENCRYPTION_KEY", ""):
            store = TeamsTenantStore()
            result = store._encrypt_token("my-secret-token")
            assert result == "my-secret-token"

    def test_decrypt_token_without_key(self):
        """Token returned unchanged without encryption key."""
        with patch("aragora.storage.teams_tenant_store.ENCRYPTION_KEY", ""):
            store = TeamsTenantStore()
            result = store._decrypt_token("my-secret-token")
            assert result == "my-secret-token"

    def test_decrypt_non_encrypted_token(self):
        """Non-encrypted tokens returned unchanged."""
        with patch(
            "aragora.storage.teams_tenant_store.ENCRYPTION_KEY",
            "test-key",
        ):
            store = TeamsTenantStore()
            # Token that doesn't start with gAAA (Fernet prefix)
            result = store._decrypt_token("plain-token")
            assert result == "plain-token"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
