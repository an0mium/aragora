"""
Tests for DiscordGuildStore - Discord guild token storage.

Tests cover:
- Guild CRUD operations
- Token encryption/decryption
- Listing and filtering
- Statistics
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.storage.discord_guild_store import (
    DiscordGuild,
    DiscordGuildStore,
    get_discord_guild_store,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_discord_guilds.db"


@pytest.fixture
def guild_store(temp_db):
    """Create a guild store with temp database."""
    return DiscordGuildStore(str(temp_db))


@pytest.fixture
def sample_guild():
    """Create a sample guild for testing."""
    return DiscordGuild(
        guild_id="123456789012345678",
        guild_name="Test Server",
        access_token="test-access-token",
        refresh_token="test-refresh-token",
        bot_user_id="bot-123",
        installed_at=time.time(),
        installed_by="user-456",
        scopes=["bot", "applications.commands"],
        tenant_id="tenant-001",
        is_active=True,
        expires_at=time.time() + 604800,  # 7 days
    )


# ===========================================================================
# DiscordGuild Dataclass Tests
# ===========================================================================


class TestDiscordGuild:
    """Tests for DiscordGuild dataclass."""

    def test_to_dict_excludes_tokens(self, sample_guild):
        """Test to_dict doesn't include sensitive tokens."""
        data = sample_guild.to_dict()

        assert "access_token" not in data
        assert "refresh_token" not in data
        assert data["guild_id"] == "123456789012345678"
        assert data["guild_name"] == "Test Server"

    def test_to_dict_includes_timestamps(self, sample_guild):
        """Test to_dict includes ISO timestamps."""
        data = sample_guild.to_dict()

        assert "installed_at" in data
        assert "installed_at_iso" in data
        assert "expires_at" in data
        assert "expires_at_iso" in data

    def test_is_token_expired_not_expired(self, sample_guild):
        """Test is_token_expired returns False for valid token."""
        sample_guild.expires_at = time.time() + 3600  # 1 hour from now
        assert sample_guild.is_token_expired() is False

    def test_is_token_expired_expired(self, sample_guild):
        """Test is_token_expired returns True for expired token."""
        sample_guild.expires_at = time.time() - 3600  # 1 hour ago
        assert sample_guild.is_token_expired() is True

    def test_is_token_expired_within_buffer(self, sample_guild):
        """Test is_token_expired returns True within 5 minute buffer."""
        sample_guild.expires_at = time.time() + 120  # 2 minutes from now (within 5 min buffer)
        assert sample_guild.is_token_expired() is True

    def test_is_token_expired_no_expiration(self, sample_guild):
        """Test is_token_expired returns False when no expiration."""
        sample_guild.expires_at = None
        assert sample_guild.is_token_expired() is False


# ===========================================================================
# DiscordGuildStore CRUD Tests
# ===========================================================================


class TestDiscordGuildStoreCRUD:
    """Tests for DiscordGuildStore CRUD operations."""

    def test_save_and_get(self, guild_store, sample_guild):
        """Test saving and retrieving a guild."""
        result = guild_store.save(sample_guild)
        assert result is True

        retrieved = guild_store.get(sample_guild.guild_id)
        assert retrieved is not None
        assert retrieved.guild_id == sample_guild.guild_id
        assert retrieved.guild_name == sample_guild.guild_name
        assert retrieved.access_token == sample_guild.access_token

    def test_get_nonexistent(self, guild_store):
        """Test getting a nonexistent guild returns None."""
        retrieved = guild_store.get("nonexistent-guild")
        assert retrieved is None

    def test_update_existing(self, guild_store, sample_guild):
        """Test updating an existing guild."""
        guild_store.save(sample_guild)

        # Modify and save again
        sample_guild.guild_name = "Updated Server Name"
        guild_store.save(sample_guild)

        retrieved = guild_store.get(sample_guild.guild_id)
        assert retrieved.guild_name == "Updated Server Name"

    def test_deactivate(self, guild_store, sample_guild):
        """Test deactivating a guild."""
        guild_store.save(sample_guild)

        result = guild_store.deactivate(sample_guild.guild_id)
        assert result is True

        retrieved = guild_store.get(sample_guild.guild_id)
        assert retrieved.is_active is False

    def test_delete(self, guild_store, sample_guild):
        """Test deleting a guild."""
        guild_store.save(sample_guild)

        result = guild_store.delete(sample_guild.guild_id)
        assert result is True

        retrieved = guild_store.get(sample_guild.guild_id)
        assert retrieved is None

    def test_update_tokens(self, guild_store, sample_guild):
        """Test updating tokens for a guild."""
        guild_store.save(sample_guild)

        new_expires = time.time() + 7200
        result = guild_store.update_tokens(
            sample_guild.guild_id,
            "new-access-token",
            "new-refresh-token",
            new_expires,
        )
        assert result is True

        retrieved = guild_store.get(sample_guild.guild_id)
        assert retrieved.access_token == "new-access-token"
        assert retrieved.refresh_token == "new-refresh-token"
        assert retrieved.expires_at == new_expires


# ===========================================================================
# DiscordGuildStore Listing Tests
# ===========================================================================


class TestDiscordGuildStoreListing:
    """Tests for DiscordGuildStore listing operations."""

    def test_list_active(self, guild_store):
        """Test listing active guilds."""
        # Create several guilds
        for i in range(5):
            guild = DiscordGuild(
                guild_id=f"guild-{i}",
                guild_name=f"Server {i}",
                access_token="token",
                bot_user_id="bot",
                installed_at=time.time(),
                is_active=i < 3,  # First 3 active
            )
            guild_store.save(guild)

        active = guild_store.list_active()
        assert len(active) == 3

    def test_list_active_pagination(self, guild_store):
        """Test listing with pagination."""
        for i in range(10):
            guild = DiscordGuild(
                guild_id=f"guild-{i}",
                guild_name=f"Server {i}",
                access_token="token",
                bot_user_id="bot",
                installed_at=time.time(),
            )
            guild_store.save(guild)

        page1 = guild_store.list_active(limit=5, offset=0)
        page2 = guild_store.list_active(limit=5, offset=5)

        assert len(page1) == 5
        assert len(page2) == 5
        assert page1[0].guild_id != page2[0].guild_id

    def test_get_by_tenant(self, guild_store):
        """Test getting guilds by tenant."""
        for i in range(5):
            guild = DiscordGuild(
                guild_id=f"guild-{i}",
                guild_name=f"Server {i}",
                access_token="token",
                bot_user_id="bot",
                installed_at=time.time(),
                tenant_id="tenant-a" if i < 3 else "tenant-b",
            )
            guild_store.save(guild)

        tenant_a_guilds = guild_store.get_by_tenant("tenant-a")
        assert len(tenant_a_guilds) == 3

    def test_list_expiring(self, guild_store):
        """Test listing guilds with expiring tokens."""
        for i in range(5):
            guild = DiscordGuild(
                guild_id=f"guild-{i}",
                guild_name=f"Server {i}",
                access_token="token",
                bot_user_id="bot",
                installed_at=time.time(),
                expires_at=time.time() + (1800 if i < 2 else 7200),  # 30 min vs 2 hours
            )
            guild_store.save(guild)

        expiring = guild_store.list_expiring(within_seconds=3600)  # 1 hour
        assert len(expiring) == 2


# ===========================================================================
# DiscordGuildStore Statistics Tests
# ===========================================================================


class TestDiscordGuildStoreStats:
    """Tests for DiscordGuildStore statistics."""

    def test_count(self, guild_store):
        """Test counting guilds."""
        for i in range(5):
            guild = DiscordGuild(
                guild_id=f"guild-{i}",
                guild_name=f"Server {i}",
                access_token="token",
                bot_user_id="bot",
                installed_at=time.time(),
                is_active=i < 3,
            )
            guild_store.save(guild)

        assert guild_store.count(active_only=True) == 3
        assert guild_store.count(active_only=False) == 5

    def test_get_stats(self, guild_store):
        """Test getting statistics."""
        for i in range(5):
            guild = DiscordGuild(
                guild_id=f"guild-{i}",
                guild_name=f"Server {i}",
                access_token="token",
                bot_user_id="bot",
                installed_at=time.time(),
                is_active=i < 3,
                expires_at=time.time() + (1800 if i < 2 else 7200),
            )
            guild_store.save(guild)

        stats = guild_store.get_stats()

        assert stats["total_guilds"] == 5
        assert stats["active_guilds"] == 3
        assert stats["inactive_guilds"] == 2


# ===========================================================================
# Singleton Tests
# ===========================================================================


class TestDiscordGuildStoreSingleton:
    """Tests for singleton behavior."""

    def test_get_discord_guild_store_returns_singleton(self):
        """Test that get_discord_guild_store returns the same instance."""
        # Reset singleton
        import aragora.storage.discord_guild_store as module

        module._guild_store = None

        store1 = get_discord_guild_store()
        store2 = get_discord_guild_store()

        assert store1 is store2


# ===========================================================================
# Token Encryption Tests
# ===========================================================================


class TestDiscordGuildStoreEncryption:
    """Tests for token encryption."""

    def test_encryption_with_key(self, temp_db):
        """Test tokens are encrypted when key is set."""
        with patch(
            "aragora.storage.discord_guild_store.ENCRYPTION_KEY",
            "test-encryption-key",
        ):
            store = DiscordGuildStore(str(temp_db))

            guild = DiscordGuild(
                guild_id="guild-encrypted",
                guild_name="Encrypted Server",
                access_token="secret-token",
                bot_user_id="bot",
                installed_at=time.time(),
            )
            store.save(guild)

            # Direct DB read should show encrypted token
            conn = store._get_connection()
            cursor = conn.execute(
                "SELECT access_token FROM discord_guilds WHERE guild_id = ?",
                (guild.guild_id,),
            )
            row = cursor.fetchone()

            # Encrypted tokens start with "gAAA" (Fernet prefix)
            assert row["access_token"] != "secret-token"

            # But getting through store should decrypt
            retrieved = store.get(guild.guild_id)
            assert retrieved.access_token == "secret-token"
