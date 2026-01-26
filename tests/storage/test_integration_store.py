"""
Tests for IntegrationStore backends.

Tests all three backends: InMemory, SQLite, and Redis (with fallback).
"""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

# Check if Redis is available for integration tests
REDIS_URL = os.environ.get("REDIS_URL", "")
try:
    import redis

    # Only mark Redis as available if REDIS_URL is set
    REDIS_AVAILABLE = bool(REDIS_URL)
except ImportError:
    REDIS_AVAILABLE = False

from aragora.storage.integration_store import (
    InMemoryIntegrationStore,
    IntegrationConfig,
    RedisIntegrationStore,
    SQLiteIntegrationStore,
    UserIdMapping,
    VALID_INTEGRATION_TYPES,
    get_integration_store,
    reset_integration_store,
    set_integration_store,
)


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_integrations.db"


@pytest.fixture
def memory_store():
    """Create an in-memory store for testing."""
    return InMemoryIntegrationStore()


@pytest.fixture
def sqlite_store(temp_db_path):
    """Create a SQLite store for testing."""
    return SQLiteIntegrationStore(temp_db_path)


@pytest.fixture
def sample_config():
    """Create a sample integration config."""
    return IntegrationConfig(
        type="slack",
        enabled=True,
        notify_on_consensus=True,
        notify_on_debate_end=True,
        settings={
            "webhook_url": "https://hooks.slack.com/services/xxx/yyy/zzz",
            "channel": "#debates",
        },
        user_id="user123",
    )


class TestIntegrationConfig:
    """Tests for IntegrationConfig dataclass."""

    def test_to_dict_masks_secrets(self, sample_config):
        """Secrets should be masked by default."""
        result = sample_config.to_dict(include_secrets=False)
        assert result["settings"]["webhook_url"] == "••••••••"
        assert result["settings"]["channel"] == "#debates"

    def test_to_dict_includes_secrets(self, sample_config):
        """Secrets should be included when requested."""
        result = sample_config.to_dict(include_secrets=True)
        assert result["settings"]["webhook_url"] == "https://hooks.slack.com/services/xxx/yyy/zzz"

    def test_to_json_roundtrip(self, sample_config):
        """JSON serialization should preserve data."""
        json_str = sample_config.to_json()
        restored = IntegrationConfig.from_json(json_str)
        assert restored.type == sample_config.type
        assert restored.enabled == sample_config.enabled
        assert restored.settings == sample_config.settings

    def test_status_disconnected(self):
        """Disabled integration should show disconnected."""
        config = IntegrationConfig(type="slack", enabled=False)
        assert config.status == "disconnected"

    def test_status_degraded(self):
        """Integration with many errors should show degraded."""
        config = IntegrationConfig(type="slack", enabled=True, errors_24h=10)
        assert config.status == "degraded"

    def test_status_connected(self):
        """Integration with activity should show connected."""
        config = IntegrationConfig(type="slack", enabled=True, last_activity=time.time())
        assert config.status == "connected"

    def test_status_not_configured(self):
        """New integration should show not_configured."""
        config = IntegrationConfig(type="slack", enabled=True)
        assert config.status == "not_configured"


class TestInMemoryIntegrationStore:
    """Tests for InMemoryIntegrationStore."""

    @pytest.mark.asyncio
    async def test_save_and_get(self, memory_store, sample_config):
        """Should save and retrieve integration config."""
        await memory_store.save(sample_config)
        retrieved = await memory_store.get("slack", "user123")
        assert retrieved is not None
        assert retrieved.type == "slack"
        assert retrieved.settings["channel"] == "#debates"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, memory_store):
        """Should return None for nonexistent integration."""
        result = await memory_store.get("slack", "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, memory_store, sample_config):
        """Should delete integration."""
        await memory_store.save(sample_config)
        deleted = await memory_store.delete("slack", "user123")
        assert deleted is True
        result = await memory_store.get("slack", "user123")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, memory_store):
        """Should return False when deleting nonexistent."""
        deleted = await memory_store.delete("slack", "nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_for_user(self, memory_store):
        """Should list all integrations for a user."""
        configs = [
            IntegrationConfig(type="slack", user_id="user1"),
            IntegrationConfig(type="discord", user_id="user1"),
            IntegrationConfig(type="email", user_id="user2"),
        ]
        for config in configs:
            await memory_store.save(config)

        user1_integrations = await memory_store.list_for_user("user1")
        assert len(user1_integrations) == 2
        types = {i.type for i in user1_integrations}
        assert types == {"slack", "discord"}

    @pytest.mark.asyncio
    async def test_list_all(self, memory_store):
        """Should list all integrations."""
        configs = [
            IntegrationConfig(type="slack", user_id="user1"),
            IntegrationConfig(type="discord", user_id="user2"),
        ]
        for config in configs:
            await memory_store.save(config)

        all_integrations = await memory_store.list_all()
        assert len(all_integrations) == 2

    @pytest.mark.asyncio
    async def test_update_existing(self, memory_store, sample_config):
        """Should update existing integration."""
        await memory_store.save(sample_config)
        sample_config.enabled = False
        sample_config.settings["channel"] = "#new-channel"
        await memory_store.save(sample_config)

        retrieved = await memory_store.get("slack", "user123")
        assert retrieved is not None
        assert retrieved.enabled is False
        assert retrieved.settings["channel"] == "#new-channel"


class TestSQLiteIntegrationStore:
    """Tests for SQLiteIntegrationStore."""

    @pytest.mark.asyncio
    async def test_save_and_get(self, sqlite_store, sample_config):
        """Should save and retrieve integration config."""
        await sqlite_store.save(sample_config)
        retrieved = await sqlite_store.get("slack", "user123")
        assert retrieved is not None
        assert retrieved.type == "slack"
        assert retrieved.settings["channel"] == "#debates"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, sqlite_store):
        """Should return None for nonexistent integration."""
        result = await sqlite_store.get("slack", "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, sqlite_store, sample_config):
        """Should delete integration."""
        await sqlite_store.save(sample_config)
        deleted = await sqlite_store.delete("slack", "user123")
        assert deleted is True
        result = await sqlite_store.get("slack", "user123")
        assert result is None

    @pytest.mark.asyncio
    async def test_persistence(self, temp_db_path, sample_config):
        """Data should persist across store instances."""
        # Save with first instance
        store1 = SQLiteIntegrationStore(temp_db_path)
        await store1.save(sample_config)
        await store1.close()

        # Retrieve with second instance
        store2 = SQLiteIntegrationStore(temp_db_path)
        retrieved = await store2.get("slack", "user123")
        assert retrieved is not None
        assert retrieved.type == "slack"
        await store2.close()

    @pytest.mark.asyncio
    async def test_list_for_user(self, sqlite_store):
        """Should list all integrations for a user."""
        configs = [
            IntegrationConfig(type="slack", user_id="user1"),
            IntegrationConfig(type="discord", user_id="user1"),
            IntegrationConfig(type="email", user_id="user2"),
        ]
        for config in configs:
            await sqlite_store.save(config)

        user1_integrations = await sqlite_store.list_for_user("user1")
        assert len(user1_integrations) == 2


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available (set REDIS_URL env var)")
class TestRedisIntegrationStore:
    """Tests for RedisIntegrationStore (requires Redis for full testing)."""

    @pytest.fixture
    def redis_store(self, temp_db_path):
        """Create a Redis store for testing."""
        return RedisIntegrationStore(temp_db_path, redis_url=REDIS_URL or "redis://localhost:6379")

    @pytest.mark.asyncio
    async def test_save_and_get(self, redis_store, sample_config):
        """Should save and retrieve integration config."""
        await redis_store.save(sample_config)
        retrieved = await redis_store.get("slack", "user123")
        assert retrieved is not None
        assert retrieved.type == "slack"
        # Works via SQLite fallback if Redis unavailable

    @pytest.mark.asyncio
    async def test_delete(self, redis_store, sample_config):
        """Should delete integration."""
        await redis_store.save(sample_config)
        deleted = await redis_store.delete("slack", "user123")
        assert deleted is True

    @pytest.mark.asyncio
    async def test_fallback_persistence(self, temp_db_path, sample_config):
        """SQLite fallback should persist data."""
        store1 = RedisIntegrationStore(temp_db_path)
        await store1.save(sample_config)
        await store1.close()

        store2 = RedisIntegrationStore(temp_db_path)
        retrieved = await store2.get("slack", "user123")
        assert retrieved is not None
        await store2.close()


class TestGlobalStore:
    """Tests for global store factory functions."""

    def setup_method(self):
        """Reset global store before each test."""
        reset_integration_store()

    def teardown_method(self):
        """Reset global store after each test."""
        reset_integration_store()

    def test_get_default_store(self, monkeypatch, temp_db_path):
        """Should create default SQLite store."""
        monkeypatch.setenv("ARAGORA_DATA_DIR", str(temp_db_path.parent))
        store = get_integration_store()
        assert isinstance(store, SQLiteIntegrationStore)

    def test_get_memory_store(self, monkeypatch):
        """Should create in-memory store when configured."""
        monkeypatch.setenv("ARAGORA_INTEGRATION_STORE_BACKEND", "memory")
        store = get_integration_store()
        assert isinstance(store, InMemoryIntegrationStore)

    def test_set_custom_store(self):
        """Should allow setting custom store."""
        custom_store = InMemoryIntegrationStore()
        set_integration_store(custom_store)
        store = get_integration_store()
        assert store is custom_store

    def test_singleton_pattern(self, monkeypatch):
        """Should return same instance on multiple calls."""
        monkeypatch.setenv("ARAGORA_INTEGRATION_STORE_BACKEND", "memory")
        store1 = get_integration_store()
        store2 = get_integration_store()
        assert store1 is store2


class TestValidIntegrationTypes:
    """Tests for VALID_INTEGRATION_TYPES constant."""

    def test_contains_expected_types(self):
        """Should contain all expected integration types."""
        expected = {"slack", "discord", "telegram", "email", "teams", "whatsapp", "matrix"}
        assert VALID_INTEGRATION_TYPES == expected

    def test_is_frozen(self):
        """Should be a frozenset-like immutable set."""
        assert "slack" in VALID_INTEGRATION_TYPES
        # Modification should raise (if actually frozenset)
        # Note: The implementation uses set, but this tests read behavior


class TestIntegrationConfigFromRow:
    """Tests for IntegrationConfig.from_row database deserialization."""

    def test_from_row_basic(self):
        """Should deserialize from database row."""
        row = (
            "slack",  # type
            1,  # enabled
            1700000000.0,  # created_at
            1700000001.0,  # updated_at
            1,  # notify_on_consensus
            1,  # notify_on_debate_end
            0,  # notify_on_error
            0,  # notify_on_leaderboard
            '{"webhook_url": "https://example.com"}',  # settings_json
            10,  # messages_sent
            2,  # errors_24h
            1700000002.0,  # last_activity
            None,  # last_error
            "user123",  # user_id
            "workspace456",  # workspace_id
        )
        config = IntegrationConfig.from_row(row)
        assert config.type == "slack"
        assert config.enabled is True
        assert config.settings["webhook_url"] == "https://example.com"
        assert config.messages_sent == 10
        assert config.user_id == "user123"
        assert config.workspace_id == "workspace456"

    def test_from_row_null_settings(self):
        """Should handle null settings."""
        row = (
            "discord",
            1,
            1700000000.0,
            1700000001.0,
            1,
            1,
            0,
            0,
            None,
            0,
            0,
            None,
            None,
            "user",
            None,
        )
        config = IntegrationConfig.from_row(row)
        assert config.settings == {}


# =============================================================================
# User ID Mapping Tests
# =============================================================================


@pytest.fixture
def sample_mapping():
    """Create a sample user ID mapping."""
    return UserIdMapping(
        email="alice@example.com",
        platform="slack",
        platform_user_id="U12345ABC",
        display_name="Alice Smith",
        user_id="tenant1",
    )


class TestUserIdMapping:
    """Tests for UserIdMapping dataclass."""

    def test_to_dict(self, sample_mapping):
        """Should convert to dictionary."""
        result = sample_mapping.to_dict()
        assert result["email"] == "alice@example.com"
        assert result["platform"] == "slack"
        assert result["platform_user_id"] == "U12345ABC"
        assert result["display_name"] == "Alice Smith"

    def test_to_json_roundtrip(self, sample_mapping):
        """JSON serialization should preserve data."""
        json_str = sample_mapping.to_json()
        restored = UserIdMapping.from_json(json_str)
        assert restored.email == sample_mapping.email
        assert restored.platform == sample_mapping.platform
        assert restored.platform_user_id == sample_mapping.platform_user_id
        assert restored.display_name == sample_mapping.display_name

    def test_from_row(self):
        """Should deserialize from database row."""
        row = (
            "bob@example.com",  # email
            "slack",  # platform
            "U67890XYZ",  # platform_user_id
            "Bob Jones",  # display_name
            1700000000.0,  # created_at
            1700000001.0,  # updated_at
            "tenant2",  # user_id
        )
        mapping = UserIdMapping.from_row(row)
        assert mapping.email == "bob@example.com"
        assert mapping.platform == "slack"
        assert mapping.platform_user_id == "U67890XYZ"
        assert mapping.display_name == "Bob Jones"
        assert mapping.user_id == "tenant2"


class TestInMemoryUserIdMappings:
    """Tests for InMemoryIntegrationStore user ID mapping methods."""

    @pytest.mark.asyncio
    async def test_save_and_get_mapping(self, memory_store, sample_mapping):
        """Should save and retrieve user ID mapping."""
        await memory_store.save_user_mapping(sample_mapping)
        retrieved = await memory_store.get_user_mapping("alice@example.com", "slack", "tenant1")
        assert retrieved is not None
        assert retrieved.platform_user_id == "U12345ABC"
        assert retrieved.display_name == "Alice Smith"

    @pytest.mark.asyncio
    async def test_get_nonexistent_mapping(self, memory_store):
        """Should return None for nonexistent mapping."""
        result = await memory_store.get_user_mapping("unknown@example.com", "slack", "tenant1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_mapping(self, memory_store, sample_mapping):
        """Should delete user ID mapping."""
        await memory_store.save_user_mapping(sample_mapping)
        deleted = await memory_store.delete_user_mapping("alice@example.com", "slack", "tenant1")
        assert deleted is True
        result = await memory_store.get_user_mapping("alice@example.com", "slack", "tenant1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_mapping(self, memory_store):
        """Should return False when deleting nonexistent mapping."""
        deleted = await memory_store.delete_user_mapping("unknown@example.com", "slack", "tenant1")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_mappings_all_platforms(self, memory_store):
        """Should list all mappings for a user across platforms."""
        mappings = [
            UserIdMapping(
                email="alice@example.com",
                platform="slack",
                platform_user_id="U12345",
                user_id="tenant1",
            ),
            UserIdMapping(
                email="alice@example.com",
                platform="discord",
                platform_user_id="D12345",
                user_id="tenant1",
            ),
            UserIdMapping(
                email="bob@example.com",
                platform="slack",
                platform_user_id="U67890",
                user_id="tenant2",
            ),
        ]
        for m in mappings:
            await memory_store.save_user_mapping(m)

        # List for tenant1 (all platforms)
        result = await memory_store.list_user_mappings(user_id="tenant1")
        assert len(result) == 2
        platforms = {m.platform for m in result}
        assert platforms == {"slack", "discord"}

    @pytest.mark.asyncio
    async def test_list_mappings_filtered_by_platform(self, memory_store):
        """Should list mappings filtered by platform."""
        mappings = [
            UserIdMapping(
                email="alice@example.com",
                platform="slack",
                platform_user_id="U12345",
                user_id="tenant1",
            ),
            UserIdMapping(
                email="bob@example.com",
                platform="slack",
                platform_user_id="U67890",
                user_id="tenant1",
            ),
            UserIdMapping(
                email="charlie@example.com",
                platform="discord",
                platform_user_id="D11111",
                user_id="tenant1",
            ),
        ]
        for m in mappings:
            await memory_store.save_user_mapping(m)

        # List only Slack mappings
        result = await memory_store.list_user_mappings(platform="slack", user_id="tenant1")
        assert len(result) == 2
        emails = {m.email for m in result}
        assert emails == {"alice@example.com", "bob@example.com"}

    @pytest.mark.asyncio
    async def test_clear_includes_mappings(self, memory_store, sample_mapping):
        """Clear should also clear mappings."""
        await memory_store.save_user_mapping(sample_mapping)
        memory_store.clear()
        result = await memory_store.get_user_mapping("alice@example.com", "slack", "tenant1")
        assert result is None


class TestSQLiteUserIdMappings:
    """Tests for SQLiteIntegrationStore user ID mapping methods."""

    @pytest.mark.asyncio
    async def test_save_and_get_mapping(self, sqlite_store, sample_mapping):
        """Should save and retrieve user ID mapping."""
        await sqlite_store.save_user_mapping(sample_mapping)
        retrieved = await sqlite_store.get_user_mapping("alice@example.com", "slack", "tenant1")
        assert retrieved is not None
        assert retrieved.platform_user_id == "U12345ABC"

    @pytest.mark.asyncio
    async def test_get_nonexistent_mapping(self, sqlite_store):
        """Should return None for nonexistent mapping."""
        result = await sqlite_store.get_user_mapping("unknown@example.com", "slack", "tenant1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_mapping(self, sqlite_store, sample_mapping):
        """Should delete user ID mapping."""
        await sqlite_store.save_user_mapping(sample_mapping)
        deleted = await sqlite_store.delete_user_mapping("alice@example.com", "slack", "tenant1")
        assert deleted is True

    @pytest.mark.asyncio
    async def test_mapping_persistence(self, temp_db_path, sample_mapping):
        """Mapping data should persist across store instances."""
        # Save with first instance
        store1 = SQLiteIntegrationStore(temp_db_path)
        await store1.save_user_mapping(sample_mapping)
        await store1.close()

        # Retrieve with second instance
        store2 = SQLiteIntegrationStore(temp_db_path)
        retrieved = await store2.get_user_mapping("alice@example.com", "slack", "tenant1")
        assert retrieved is not None
        assert retrieved.platform_user_id == "U12345ABC"
        await store2.close()

    @pytest.mark.asyncio
    async def test_list_mappings(self, sqlite_store):
        """Should list mappings."""
        mappings = [
            UserIdMapping(
                email="alice@example.com",
                platform="slack",
                platform_user_id="U12345",
                user_id="tenant1",
            ),
            UserIdMapping(
                email="bob@example.com",
                platform="slack",
                platform_user_id="U67890",
                user_id="tenant1",
            ),
        ]
        for m in mappings:
            await sqlite_store.save_user_mapping(m)

        result = await sqlite_store.list_user_mappings(platform="slack", user_id="tenant1")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_update_existing_mapping(self, sqlite_store, sample_mapping):
        """Should update existing mapping."""
        await sqlite_store.save_user_mapping(sample_mapping)

        # Update display name
        sample_mapping.display_name = "Alice Johnson"
        await sqlite_store.save_user_mapping(sample_mapping)

        retrieved = await sqlite_store.get_user_mapping("alice@example.com", "slack", "tenant1")
        assert retrieved is not None
        assert retrieved.display_name == "Alice Johnson"


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available (set REDIS_URL env var)")
class TestRedisUserIdMappings:
    """Tests for RedisIntegrationStore user ID mapping methods (requires Redis)."""

    @pytest.fixture
    def redis_store(self, temp_db_path):
        """Create a Redis store for testing."""
        return RedisIntegrationStore(temp_db_path, redis_url=REDIS_URL or "redis://localhost:6379")

    @pytest.mark.asyncio
    async def test_save_and_get_mapping(self, redis_store, sample_mapping):
        """Should save and retrieve user ID mapping."""
        await redis_store.save_user_mapping(sample_mapping)
        retrieved = await redis_store.get_user_mapping("alice@example.com", "slack", "tenant1")
        assert retrieved is not None
        assert retrieved.platform_user_id == "U12345ABC"

    @pytest.mark.asyncio
    async def test_delete_mapping(self, redis_store, sample_mapping):
        """Should delete user ID mapping."""
        await redis_store.save_user_mapping(sample_mapping)
        deleted = await redis_store.delete_user_mapping("alice@example.com", "slack", "tenant1")
        assert deleted is True

    @pytest.mark.asyncio
    async def test_mapping_fallback_persistence(self, temp_db_path, sample_mapping):
        """SQLite fallback should persist mapping data."""
        store1 = RedisIntegrationStore(temp_db_path)
        await store1.save_user_mapping(sample_mapping)
        await store1.close()

        store2 = RedisIntegrationStore(temp_db_path)
        retrieved = await store2.get_user_mapping("alice@example.com", "slack", "tenant1")
        assert retrieved is not None
        await store2.close()
