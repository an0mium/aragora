"""
Tests for session store utilities.

Tests cover:
- SessionStoreConfig dataclass
- InMemorySessionStore class
- get_session_store / reset_session_store functions
"""

import pytest
import threading
from unittest.mock import patch, MagicMock

from aragora.server.session_store import (
    SessionStoreConfig,
    InMemorySessionStore,
    get_session_store,
    reset_session_store,
)


@pytest.fixture(autouse=True)
def reset_store():
    """Reset global session store before and after each test."""
    reset_session_store()
    yield
    reset_session_store()


class TestSessionStoreConfig:
    """Tests for SessionStoreConfig dataclass."""

    def test_default_values(self):
        """Config has correct defaults."""
        config = SessionStoreConfig()

        assert config.debate_state_ttl > 0
        assert config.active_loop_ttl > 0
        assert config.auth_state_ttl > 0
        assert config.rate_limiter_ttl > 0
        assert config.max_debate_states > 0
        assert config.max_active_loops > 0
        assert config.max_auth_states > 0
        assert config.key_prefix == "aragora:session:"

    def test_custom_values(self):
        """Config accepts custom values."""
        config = SessionStoreConfig(
            debate_state_ttl=100,
            max_debate_states=10,
            key_prefix="test:",
        )

        assert config.debate_state_ttl == 100
        assert config.max_debate_states == 10
        assert config.key_prefix == "test:"


class TestInMemorySessionStore:
    """Tests for InMemorySessionStore class."""

    @pytest.fixture
    def store(self):
        """Fresh in-memory store for each test."""
        config = SessionStoreConfig(
            debate_state_ttl=1,  # 1 second for fast expiry tests
            active_loop_ttl=1,
            auth_state_ttl=1,
            max_debate_states=5,
            max_active_loops=5,
            max_auth_states=5,
        )
        return InMemorySessionStore(config)

    def test_is_not_distributed(self, store):
        """In-memory store is not distributed."""
        assert store.is_distributed is False

    # Debate state tests
    def test_set_and_get_debate_state(self, store):
        """Sets and gets debate state."""
        store.set_debate_state("loop-1", {"status": "running"})
        state = store.get_debate_state("loop-1")

        assert state is not None
        assert state["status"] == "running"

    def test_get_nonexistent_debate_state(self, store):
        """Returns None for nonexistent state."""
        assert store.get_debate_state("nonexistent") is None

    def test_debate_state_returns_copy(self, store):
        """Returns a copy, not the original."""
        original = {"status": "running"}
        store.set_debate_state("loop-1", original)
        retrieved = store.get_debate_state("loop-1")

        retrieved["status"] = "modified"
        assert store.get_debate_state("loop-1")["status"] == "running"

    def test_delete_debate_state(self, store):
        """Deletes debate state."""
        store.set_debate_state("loop-1", {"status": "running"})
        assert store.delete_debate_state("loop-1") is True
        assert store.get_debate_state("loop-1") is None

    def test_delete_nonexistent_debate_state(self, store):
        """Returns False when deleting nonexistent state."""
        assert store.delete_debate_state("nonexistent") is False

    def test_debate_state_lru_eviction(self, store):
        """Evicts oldest debate states when max reached."""
        for i in range(6):
            store.set_debate_state(f"loop-{i}", {"num": i})

        # First one should be evicted
        assert store.get_debate_state("loop-0") is None
        assert store.get_debate_state("loop-5") is not None

    # Active loop tests
    def test_set_and_get_active_loop(self, store):
        """Sets and gets active loop."""
        store.set_active_loop("loop-1", {"phase": "debate"})
        data = store.get_active_loop("loop-1")

        assert data is not None
        assert data["phase"] == "debate"

    def test_get_nonexistent_active_loop(self, store):
        """Returns None for nonexistent loop."""
        assert store.get_active_loop("nonexistent") is None

    def test_delete_active_loop(self, store):
        """Deletes active loop."""
        store.set_active_loop("loop-1", {"phase": "debate"})
        assert store.delete_active_loop("loop-1") is True
        assert store.get_active_loop("loop-1") is None

    def test_list_active_loops(self, store):
        """Lists all active loop IDs."""
        store.set_active_loop("loop-1", {})
        store.set_active_loop("loop-2", {})
        store.set_active_loop("loop-3", {})

        loops = store.list_active_loops()
        assert set(loops) == {"loop-1", "loop-2", "loop-3"}

    def test_active_loop_lru_eviction(self, store):
        """Evicts oldest active loops when max reached."""
        for i in range(6):
            store.set_active_loop(f"loop-{i}", {"num": i})

        # First one should be evicted
        assert store.get_active_loop("loop-0") is None
        assert store.get_active_loop("loop-5") is not None

    # Auth state tests
    def test_set_and_get_auth_state(self, store):
        """Sets and gets auth state."""
        store.set_auth_state("conn-1", {"user_id": "u1"})
        state = store.get_auth_state("conn-1")

        assert state is not None
        assert state["user_id"] == "u1"

    def test_get_nonexistent_auth_state(self, store):
        """Returns None for nonexistent auth state."""
        assert store.get_auth_state("nonexistent") is None

    def test_delete_auth_state(self, store):
        """Deletes auth state."""
        store.set_auth_state("conn-1", {"user_id": "u1"})
        assert store.delete_auth_state("conn-1") is True
        assert store.get_auth_state("conn-1") is None

    def test_auth_state_lru_eviction(self, store):
        """Evicts oldest auth states when max reached."""
        for i in range(6):
            store.set_auth_state(f"conn-{i}", {"num": i})

        # First one should be evicted
        assert store.get_auth_state("conn-0") is None
        assert store.get_auth_state("conn-5") is not None

    # Pub/Sub tests
    def test_publish_and_subscribe(self, store):
        """Publishes and subscribes to events."""
        received = []
        store.subscribe_events("test-channel", lambda e: received.append(e))

        store.publish_event("test-channel", {"type": "test"})

        assert len(received) == 1
        assert received[0]["type"] == "test"

    def test_multiple_subscribers(self, store):
        """Multiple subscribers receive events."""
        received1 = []
        received2 = []
        store.subscribe_events("channel", lambda e: received1.append(e))
        store.subscribe_events("channel", lambda e: received2.append(e))

        store.publish_event("channel", {"data": 1})

        assert len(received1) == 1
        assert len(received2) == 1

    def test_subscriber_error_doesnt_break_others(self, store):
        """One subscriber error doesn't affect others."""
        received = []

        def failing_handler(e):
            raise ValueError("fail")

        store.subscribe_events("channel", failing_handler)
        store.subscribe_events("channel", lambda e: received.append(e))

        store.publish_event("channel", {"data": 1})
        assert len(received) == 1

    # Cleanup tests
    def test_cleanup_expired(self, store, monkeypatch):
        """Cleans up expired entries."""
        now = 1_000_000.0
        monkeypatch.setattr("aragora.server.session_store.time.time", lambda: now)

        store.set_debate_state("loop-1", {"status": "old"})
        store.set_active_loop("loop-1", {"phase": "old"})
        store.set_auth_state("conn-1", {"user_id": "old"})

        # Advance time beyond TTL (TTL is 1 second)
        now += 2.0

        counts = store.cleanup_expired()

        assert counts["debate_states"] == 1
        assert counts["active_loops"] == 1
        assert counts["auth_states"] == 1

        assert store.get_debate_state("loop-1") is None
        assert store.get_active_loop("loop-1") is None
        assert store.get_auth_state("conn-1") is None

    def test_cleanup_preserves_fresh_entries(self, store):
        """Cleanup preserves non-expired entries."""
        store.set_debate_state("loop-1", {"status": "fresh"})

        counts = store.cleanup_expired()

        assert counts["debate_states"] == 0
        assert store.get_debate_state("loop-1") is not None


class TestInMemorySessionStoreThreadSafety:
    """Thread safety tests for InMemorySessionStore."""

    def test_concurrent_debate_state_operations(self):
        """Handles concurrent debate state operations."""
        store = InMemorySessionStore(SessionStoreConfig(max_debate_states=100))
        errors = []

        def worker(n):
            try:
                for i in range(10):
                    store.set_debate_state(f"loop-{n}-{i}", {"n": n, "i": i})
                    store.get_debate_state(f"loop-{n}-{i}")
                    if i % 2 == 0:
                        store.delete_debate_state(f"loop-{n}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_pubsub(self):
        """Handles concurrent pub/sub."""
        store = InMemorySessionStore()
        received = []
        lock = threading.Lock()

        def handler(e):
            with lock:
                received.append(e)

        store.subscribe_events("channel", handler)

        def publisher(n):
            for i in range(10):
                store.publish_event("channel", {"n": n, "i": i})

        threads = [threading.Thread(target=publisher, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(received) == 50


class TestGetSessionStore:
    """Tests for get_session_store function."""

    def test_returns_inmemory_store(self):
        """Returns in-memory store when Redis unavailable."""
        store = get_session_store(force_memory=True)
        assert store.is_distributed is False
        assert isinstance(store, InMemorySessionStore)

    def test_returns_singleton(self):
        """Returns same instance on multiple calls."""
        store1 = get_session_store(force_memory=True)
        store2 = get_session_store(force_memory=True)
        assert store1 is store2

    def test_force_memory_skips_redis(self):
        """force_memory=True skips Redis check."""
        # With force_memory=True, Redis is never checked
        store = get_session_store(force_memory=True)
        assert store.is_distributed is False

    def test_falls_back_to_memory_on_redis_error(self):
        """Falls back to in-memory when Redis fails."""
        reset_session_store()
        # The import of is_redis_available happens inside get_session_store
        # Patch at the source module
        with patch("aragora.server.redis_config.is_redis_available") as mock_check:
            mock_check.side_effect = Exception("Redis connection failed")

            store = get_session_store()
            assert store.is_distributed is False


class TestResetSessionStore:
    """Tests for reset_session_store function."""

    def test_clears_singleton(self):
        """Clears the singleton instance."""
        store1 = get_session_store(force_memory=True)
        reset_session_store()
        store2 = get_session_store(force_memory=True)

        # Should be different instances
        assert store1 is not store2

    def test_calls_close_if_available(self):
        """Calls close() on store if available."""
        store = get_session_store(force_memory=True)
        # Add a close method
        store.close = MagicMock()

        reset_session_store()
        store.close.assert_called_once()


class TestEdgeCases:
    """Edge case tests."""

    def test_store_with_complex_data(self):
        """Handles complex nested data."""
        store = InMemorySessionStore()
        complex_data = {
            "nested": {
                "list": [1, 2, {"deep": True}],
                "dict": {"a": 1, "b": 2},
            },
            "unicode": "🎉 café",
            "none": None,
        }

        store.set_debate_state("loop-1", complex_data)
        retrieved = store.get_debate_state("loop-1")

        assert retrieved["nested"]["list"][2]["deep"] is True
        assert retrieved["unicode"] == "🎉 café"
        assert retrieved["none"] is None

    def test_empty_data(self):
        """Handles empty data."""
        store = InMemorySessionStore()

        store.set_debate_state("loop-1", {})
        store.set_active_loop("loop-1", {})
        store.set_auth_state("conn-1", {})

        assert store.get_debate_state("loop-1") == {}
        assert store.get_active_loop("loop-1") == {}
        assert store.get_auth_state("conn-1") == {}

    def test_overwrite_existing(self):
        """Overwrites existing entries."""
        store = InMemorySessionStore()

        store.set_debate_state("loop-1", {"version": 1})
        store.set_debate_state("loop-1", {"version": 2})

        assert store.get_debate_state("loop-1")["version"] == 2

    def test_access_updates_lru_order(self):
        """Accessing an entry updates LRU order."""
        config = SessionStoreConfig(max_debate_states=3)
        store = InMemorySessionStore(config)

        store.set_debate_state("loop-1", {})
        store.set_debate_state("loop-2", {})
        store.set_debate_state("loop-3", {})

        # Access loop-1 to make it "recent"
        store.get_debate_state("loop-1")

        # Add loop-4, should evict loop-2 (oldest unused)
        store.set_debate_state("loop-4", {})

        assert store.get_debate_state("loop-1") is not None
        assert store.get_debate_state("loop-2") is None
        assert store.get_debate_state("loop-3") is not None
        assert store.get_debate_state("loop-4") is not None
