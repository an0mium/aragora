"""
Tests for WebhookConfigStore - durable webhook configuration storage.

Tests cover:
- InMemoryWebhookConfigStore
- SQLiteWebhookConfigStore
- RedisWebhookConfigStore (with SQLite fallback)
- Factory function get_webhook_config_store
"""

import os
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from aragora.storage.webhook_config_store import (
    WebhookConfig,
    WebhookConfigStoreBackend,
    InMemoryWebhookConfigStore,
    SQLiteWebhookConfigStore,
    RedisWebhookConfigStore,
    get_webhook_config_store,
    set_webhook_config_store,
    reset_webhook_config_store,
    WEBHOOK_EVENTS,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def in_memory_store():
    """Create fresh in-memory store for testing."""
    return InMemoryWebhookConfigStore()


@pytest.fixture
def sqlite_store(tmp_path):
    """Create fresh SQLite store for testing."""
    db_path = tmp_path / "test_webhooks.db"
    store = SQLiteWebhookConfigStore(db_path)
    yield store
    store.close()


@pytest.fixture
def redis_store(tmp_path):
    """Create fresh Redis store (falls back to SQLite) for testing."""
    db_path = tmp_path / "test_webhooks_redis.db"
    store = RedisWebhookConfigStore(db_path, redis_url="redis://nonexistent:6379")
    yield store
    store.close()


@pytest.fixture(autouse=True)
def reset_global_store():
    """Reset global store between tests."""
    reset_webhook_config_store()
    yield
    reset_webhook_config_store()


# =============================================================================
# WebhookConfig Model Tests
# =============================================================================


class TestWebhookConfig:
    """Tests for WebhookConfig dataclass."""

    def test_create_minimal_config(self):
        """Test creating a config with minimal required fields."""
        config = WebhookConfig(
            id="test-id",
            url="https://example.com/webhook",
            events=["debate_end"],
            secret="test-secret",
        )
        assert config.id == "test-id"
        assert config.url == "https://example.com/webhook"
        assert config.events == ["debate_end"]
        assert config.secret == "test-secret"
        assert config.active is True
        assert config.name is None

    def test_to_dict_excludes_secret(self):
        """Test to_dict excludes secret by default."""
        config = WebhookConfig(
            id="test-id",
            url="https://example.com",
            events=["debate_end"],
            secret="secret-value",
        )
        result = config.to_dict()
        assert "secret" not in result
        assert result["id"] == "test-id"

    def test_to_dict_includes_secret(self):
        """Test to_dict can include secret when requested."""
        config = WebhookConfig(
            id="test-id",
            url="https://example.com",
            events=["debate_end"],
            secret="secret-value",
        )
        result = config.to_dict(include_secret=True)
        assert result["secret"] == "secret-value"

    def test_matches_event_active(self):
        """Test matches_event for active webhook."""
        config = WebhookConfig(
            id="test-id",
            url="https://example.com",
            events=["debate_start", "debate_end"],
            secret="secret",
            active=True,
        )
        assert config.matches_event("debate_start") is True
        assert config.matches_event("debate_end") is True
        assert config.matches_event("vote") is False

    def test_matches_event_inactive(self):
        """Test matches_event returns False for inactive webhook."""
        config = WebhookConfig(
            id="test-id",
            url="https://example.com",
            events=["debate_start"],
            secret="secret",
            active=False,
        )
        assert config.matches_event("debate_start") is False

    def test_matches_event_wildcard(self):
        """Test matches_event with wildcard subscription."""
        config = WebhookConfig(
            id="test-id",
            url="https://example.com",
            events=["*"],
            secret="secret",
            active=True,
        )
        assert config.matches_event("debate_start") is True
        assert config.matches_event("vote") is True
        assert config.matches_event("invalid_event") is False  # Not in WEBHOOK_EVENTS

    def test_to_json_roundtrip(self):
        """Test JSON serialization roundtrip."""
        config = WebhookConfig(
            id="test-id",
            url="https://example.com",
            events=["debate_end"],
            secret="secret",
            name="Test",
            user_id="user-1",
        )
        json_str = config.to_json()
        restored = WebhookConfig.from_json(json_str)
        assert restored.id == config.id
        assert restored.url == config.url
        assert restored.events == config.events
        assert restored.name == config.name
        assert restored.user_id == config.user_id


# =============================================================================
# InMemoryWebhookConfigStore Tests
# =============================================================================


class TestInMemoryWebhookConfigStore:
    """Tests for InMemoryWebhookConfigStore."""

    def test_register_webhook(self, in_memory_store):
        """Test registering a new webhook."""
        webhook = in_memory_store.register(
            url="https://example.com/hook",
            events=["debate_end"],
            name="Test Hook",
        )
        assert webhook.id is not None
        assert webhook.url == "https://example.com/hook"
        assert webhook.events == ["debate_end"]
        assert webhook.name == "Test Hook"
        assert webhook.secret is not None
        assert len(webhook.secret) > 20  # Should be a secure token

    def test_get_webhook(self, in_memory_store):
        """Test retrieving a webhook by ID."""
        created = in_memory_store.register(
            url="https://example.com/hook",
            events=["debate_end"],
        )
        retrieved = in_memory_store.get(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.url == created.url

    def test_get_nonexistent_webhook(self, in_memory_store):
        """Test retrieving a nonexistent webhook."""
        result = in_memory_store.get("nonexistent-id")
        assert result is None

    def test_list_webhooks(self, in_memory_store):
        """Test listing all webhooks."""
        in_memory_store.register(url="https://a.com", events=["debate_end"])
        in_memory_store.register(url="https://b.com", events=["vote"])
        webhooks = in_memory_store.list()
        assert len(webhooks) == 2

    def test_list_webhooks_by_user(self, in_memory_store):
        """Test listing webhooks filtered by user."""
        in_memory_store.register(url="https://a.com", events=["debate_end"], user_id="user-1")
        in_memory_store.register(url="https://b.com", events=["vote"], user_id="user-2")
        webhooks = in_memory_store.list(user_id="user-1")
        assert len(webhooks) == 1
        assert webhooks[0].user_id == "user-1"

    def test_list_webhooks_active_only(self, in_memory_store):
        """Test listing only active webhooks."""
        w1 = in_memory_store.register(url="https://a.com", events=["debate_end"])
        in_memory_store.register(url="https://b.com", events=["vote"])
        in_memory_store.update(w1.id, active=False)
        webhooks = in_memory_store.list(active_only=True)
        assert len(webhooks) == 1
        assert webhooks[0].active is True

    def test_delete_webhook(self, in_memory_store):
        """Test deleting a webhook."""
        webhook = in_memory_store.register(url="https://a.com", events=["debate_end"])
        result = in_memory_store.delete(webhook.id)
        assert result is True
        assert in_memory_store.get(webhook.id) is None

    def test_delete_nonexistent_webhook(self, in_memory_store):
        """Test deleting a nonexistent webhook."""
        result = in_memory_store.delete("nonexistent-id")
        assert result is False

    def test_update_webhook(self, in_memory_store):
        """Test updating a webhook."""
        webhook = in_memory_store.register(
            url="https://old.com",
            events=["debate_end"],
            name="Old Name",
        )
        updated = in_memory_store.update(
            webhook.id,
            url="https://new.com",
            name="New Name",
        )
        assert updated.url == "https://new.com"
        assert updated.name == "New Name"
        assert updated.updated_at > webhook.created_at

    def test_update_webhook_partial(self, in_memory_store):
        """Test partial update of webhook."""
        webhook = in_memory_store.register(
            url="https://example.com",
            events=["debate_end"],
            name="Original Name",
        )
        updated = in_memory_store.update(webhook.id, name="New Name")
        assert updated.url == "https://example.com"  # Unchanged
        assert updated.name == "New Name"

    def test_record_delivery_success(self, in_memory_store):
        """Test recording successful delivery."""
        webhook = in_memory_store.register(url="https://a.com", events=["debate_end"])
        in_memory_store.record_delivery(webhook.id, 200, success=True)
        updated = in_memory_store.get(webhook.id)
        assert updated.last_delivery_status == 200
        assert updated.delivery_count == 1
        assert updated.failure_count == 0

    def test_record_delivery_failure(self, in_memory_store):
        """Test recording failed delivery."""
        webhook = in_memory_store.register(url="https://a.com", events=["debate_end"])
        in_memory_store.record_delivery(webhook.id, 500, success=False)
        updated = in_memory_store.get(webhook.id)
        assert updated.last_delivery_status == 500
        assert updated.delivery_count == 1
        assert updated.failure_count == 1

    def test_get_for_event(self, in_memory_store):
        """Test getting webhooks for a specific event."""
        in_memory_store.register(url="https://a.com", events=["debate_end", "vote"])
        in_memory_store.register(url="https://b.com", events=["consensus"])
        webhooks = in_memory_store.get_for_event("debate_end")
        assert len(webhooks) == 1
        assert "debate_end" in webhooks[0].events

    def test_clear(self, in_memory_store):
        """Test clearing all webhooks."""
        in_memory_store.register(url="https://a.com", events=["debate_end"])
        in_memory_store.register(url="https://b.com", events=["vote"])
        in_memory_store.clear()
        assert len(in_memory_store.list()) == 0


# =============================================================================
# SQLiteWebhookConfigStore Tests
# =============================================================================


class TestSQLiteWebhookConfigStore:
    """Tests for SQLiteWebhookConfigStore."""

    def test_register_and_get(self, sqlite_store):
        """Test registering and retrieving a webhook."""
        webhook = sqlite_store.register(
            url="https://example.com/hook",
            events=["debate_end"],
            name="Test Hook",
        )
        assert webhook.id is not None
        retrieved = sqlite_store.get(webhook.id)
        assert retrieved is not None
        assert retrieved.url == "https://example.com/hook"

    def test_persistence(self, tmp_path):
        """Test that data persists across store instances."""
        db_path = tmp_path / "persist_test.db"

        # Create and register
        store1 = SQLiteWebhookConfigStore(db_path)
        webhook = store1.register(url="https://a.com", events=["debate_end"])
        webhook_id = webhook.id
        store1.close()

        # Open new instance and verify data persists
        store2 = SQLiteWebhookConfigStore(db_path)
        retrieved = store2.get(webhook_id)
        assert retrieved is not None
        assert retrieved.url == "https://a.com"
        store2.close()

    def test_list_webhooks(self, sqlite_store):
        """Test listing all webhooks."""
        sqlite_store.register(url="https://a.com", events=["debate_end"])
        sqlite_store.register(url="https://b.com", events=["vote"])
        webhooks = sqlite_store.list()
        assert len(webhooks) == 2

    def test_list_by_user(self, sqlite_store):
        """Test listing webhooks by user."""
        sqlite_store.register(url="https://a.com", events=["debate_end"], user_id="user-1")
        sqlite_store.register(url="https://b.com", events=["vote"], user_id="user-2")
        webhooks = sqlite_store.list(user_id="user-1")
        assert len(webhooks) == 1

    def test_delete_webhook(self, sqlite_store):
        """Test deleting a webhook."""
        webhook = sqlite_store.register(url="https://a.com", events=["debate_end"])
        result = sqlite_store.delete(webhook.id)
        assert result is True
        assert sqlite_store.get(webhook.id) is None

    def test_update_webhook(self, sqlite_store):
        """Test updating a webhook."""
        webhook = sqlite_store.register(url="https://old.com", events=["debate_end"])
        updated = sqlite_store.update(webhook.id, url="https://new.com")
        assert updated.url == "https://new.com"
        # Verify persistence
        retrieved = sqlite_store.get(webhook.id)
        assert retrieved.url == "https://new.com"

    def test_record_delivery(self, sqlite_store):
        """Test recording delivery status."""
        webhook = sqlite_store.register(url="https://a.com", events=["debate_end"])
        sqlite_store.record_delivery(webhook.id, 200, success=True)
        updated = sqlite_store.get(webhook.id)
        assert updated.last_delivery_status == 200
        assert updated.delivery_count == 1

    def test_get_for_event(self, sqlite_store):
        """Test getting webhooks for an event."""
        sqlite_store.register(url="https://a.com", events=["debate_end"])
        sqlite_store.register(url="https://b.com", events=["vote"])
        webhooks = sqlite_store.get_for_event("debate_end")
        assert len(webhooks) == 1


# =============================================================================
# RedisWebhookConfigStore Tests (with SQLite fallback)
# =============================================================================


class TestRedisWebhookConfigStore:
    """Tests for RedisWebhookConfigStore (falls back to SQLite when Redis unavailable)."""

    def test_fallback_to_sqlite(self, redis_store):
        """Test that store works when Redis is unavailable (using SQLite fallback)."""
        webhook = redis_store.register(
            url="https://example.com/hook",
            events=["debate_end"],
        )
        assert webhook.id is not None
        retrieved = redis_store.get(webhook.id)
        assert retrieved is not None
        assert retrieved.url == "https://example.com/hook"

    def test_list_uses_sqlite(self, redis_store):
        """Test that list operations use SQLite backend."""
        redis_store.register(url="https://a.com", events=["debate_end"])
        redis_store.register(url="https://b.com", events=["vote"])
        webhooks = redis_store.list()
        assert len(webhooks) == 2

    def test_delete(self, redis_store):
        """Test deleting a webhook."""
        webhook = redis_store.register(url="https://a.com", events=["debate_end"])
        result = redis_store.delete(webhook.id)
        assert result is True
        assert redis_store.get(webhook.id) is None


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestGetWebhookConfigStore:
    """Tests for get_webhook_config_store factory function."""

    def test_default_is_sqlite(self, tmp_path):
        """Test that default backend is SQLite."""
        with patch.dict(os.environ, {"ARAGORA_DATA_DIR": str(tmp_path)}, clear=False):
            reset_webhook_config_store()
            store = get_webhook_config_store()
            assert isinstance(store, SQLiteWebhookConfigStore)

    def test_memory_backend(self, tmp_path):
        """Test in-memory backend selection."""
        with patch.dict(
            os.environ,
            {"ARAGORA_WEBHOOK_CONFIG_STORE_BACKEND": "memory", "ARAGORA_DATA_DIR": str(tmp_path)},
            clear=False,
        ):
            reset_webhook_config_store()
            store = get_webhook_config_store()
            assert isinstance(store, InMemoryWebhookConfigStore)

    def test_singleton_behavior(self, tmp_path):
        """Test that factory returns singleton."""
        with patch.dict(os.environ, {"ARAGORA_DATA_DIR": str(tmp_path)}, clear=False):
            reset_webhook_config_store()
            store1 = get_webhook_config_store()
            store2 = get_webhook_config_store()
            assert store1 is store2

    def test_set_webhook_config_store(self):
        """Test setting a custom store."""
        custom_store = InMemoryWebhookConfigStore()
        set_webhook_config_store(custom_store)
        assert get_webhook_config_store() is custom_store


# =============================================================================
# WEBHOOK_EVENTS Tests
# =============================================================================


class TestWebhookEvents:
    """Tests for WEBHOOK_EVENTS set."""

    def test_events_not_empty(self):
        """Test that WEBHOOK_EVENTS is not empty."""
        assert len(WEBHOOK_EVENTS) > 0

    def test_contains_core_events(self):
        """Test that core events are present."""
        assert "debate_start" in WEBHOOK_EVENTS
        assert "debate_end" in WEBHOOK_EVENTS
        assert "consensus" in WEBHOOK_EVENTS
        assert "vote" in WEBHOOK_EVENTS

    def test_all_events_are_strings(self):
        """Test that all events are strings."""
        for event in WEBHOOK_EVENTS:
            assert isinstance(event, str)
            assert len(event) > 0
