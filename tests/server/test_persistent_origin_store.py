"""
Tests for PersistentOriginStore.
"""

import pytest
import time
from pathlib import Path

from aragora.server.persistent_origin_store import (
    PersistentOriginStore,
    OriginRecord,
    reset_origin_store,
)


class TestOriginRecord:
    """Tests for OriginRecord dataclass."""

    def test_creation(self):
        """Should create record with defaults."""
        record = OriginRecord(
            origin_id="test-1",
            origin_type="debate",
            platform="slack",
            channel_id="C1234",
            user_id="U5678",
        )
        assert record.origin_id == "test-1"
        assert record.origin_type == "debate"
        assert record.platform == "slack"
        assert record.result_sent is False
        assert record.expires_at is not None

    def test_is_expired(self):
        """Should detect expired records."""
        # Not expired
        record = OriginRecord(
            origin_id="test-1",
            origin_type="debate",
            platform="slack",
            channel_id="C1234",
            user_id="U5678",
            expires_at=time.time() + 3600,
        )
        assert record.is_expired() is False

        # Expired
        expired = OriginRecord(
            origin_id="test-2",
            origin_type="debate",
            platform="slack",
            channel_id="C1234",
            user_id="U5678",
            expires_at=time.time() - 1,
        )
        assert expired.is_expired() is True

    def test_to_dict_roundtrip(self):
        """Should serialize and deserialize."""
        record = OriginRecord(
            origin_id="test-1",
            origin_type="email_reply",
            platform="email",
            channel_id="test@example.com",
            user_id="sender@example.com",
            metadata={"subject": "Test", "message_id": "<abc@xyz>"},
            thread_id="thread-123",
        )

        data = record.to_dict()
        restored = OriginRecord.from_dict(data)

        assert restored.origin_id == record.origin_id
        assert restored.origin_type == record.origin_type
        assert restored.platform == record.platform
        assert restored.metadata == record.metadata
        assert restored.thread_id == record.thread_id


class TestPersistentOriginStoreSQLite:
    """Tests for PersistentOriginStore with SQLite backend."""

    @pytest.fixture
    async def store(self, tmp_path):
        """Create a store with temp SQLite database."""
        db_path = tmp_path / "test_origins.db"
        store = PersistentOriginStore()
        store._sqlite_path = str(db_path)
        store._use_postgres = False
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_initialize(self, tmp_path):
        """Should initialize SQLite database."""
        db_path = tmp_path / "init_test.db"
        store = PersistentOriginStore()
        store._sqlite_path = str(db_path)
        store._use_postgres = False
        await store.initialize()
        assert store._initialized is True
        assert store._use_postgres is False
        await store.close()

    @pytest.mark.asyncio
    async def test_register_and_get_origin(self, store):
        """Should register and retrieve origins."""
        origin = await store.register_origin(
            origin_id="debate-123",
            origin_type="debate",
            platform="telegram",
            channel_id="12345678",
            user_id="87654321",
            thread_id="thread-1",
            metadata={"username": "testuser"},
        )

        assert origin.origin_id == "debate-123"
        assert origin.platform == "telegram"

        # Retrieve
        retrieved = await store.get_origin("debate-123")
        assert retrieved is not None
        assert retrieved.origin_id == "debate-123"
        assert retrieved.metadata["username"] == "testuser"

    @pytest.mark.asyncio
    async def test_get_nonexistent_origin(self, store):
        """Should return None for nonexistent origin."""
        result = await store.get_origin("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_mark_result_sent(self, store):
        """Should mark result as sent."""
        await store.register_origin(
            origin_id="debate-456",
            origin_type="debate",
            platform="slack",
            channel_id="C1234",
            user_id="U5678",
        )

        # Mark as sent
        success = await store.mark_result_sent(
            "debate-456",
            result_data={"status": "completed"},
        )
        assert success is True

        # Verify
        origin = await store.get_origin("debate-456")
        assert origin.result_sent is True
        assert origin.result_sent_at is not None
        assert origin.metadata["result"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_list_pending(self, tmp_path):
        """Should list pending origins."""
        db_path = tmp_path / "list_pending.db"
        store = PersistentOriginStore()
        store._sqlite_path = str(db_path)
        store._use_postgres = False
        await store.initialize()

        try:
            await store.register_origin(
                origin_id="pending-1",
                origin_type="debate",
                platform="slack",
                channel_id="C1",
                user_id="U1",
            )
            await store.register_origin(
                origin_id="pending-2",
                origin_type="email_reply",
                platform="email",
                channel_id="test@test.com",
                user_id="sender@test.com",
            )

            # Mark one as sent
            await store.mark_result_sent("pending-1")

            # List pending
            pending = await store.list_pending()
            assert len(pending) == 1
            assert pending[0].origin_id == "pending-2"
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_list_pending_with_filters(self, tmp_path):
        """Should filter pending by type and platform."""
        db_path = tmp_path / "list_filters.db"
        store = PersistentOriginStore()
        store._sqlite_path = str(db_path)
        store._use_postgres = False
        await store.initialize()

        try:
            await store.register_origin(
                origin_id="o1",
                origin_type="debate",
                platform="slack",
                channel_id="C1",
                user_id="U1",
            )
            await store.register_origin(
                origin_id="o2",
                origin_type="debate",
                platform="telegram",
                channel_id="123",
                user_id="456",
            )
            await store.register_origin(
                origin_id="o3",
                origin_type="email_reply",
                platform="email",
                channel_id="a@b.com",
                user_id="c@d.com",
            )

            # Filter by type
            debates = await store.list_pending(origin_type="debate")
            assert len(debates) == 2

            # Filter by platform
            slack = await store.list_pending(platform="slack")
            assert len(slack) == 1
            assert slack[0].platform == "slack"
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, store):
        """Should remove expired origins."""
        # Create expired origin
        await store.register_origin(
            origin_id="expired-1",
            origin_type="debate",
            platform="slack",
            channel_id="C1",
            user_id="U1",
            ttl_seconds=-1,  # Already expired
        )

        # Create valid origin
        await store.register_origin(
            origin_id="valid-1",
            origin_type="debate",
            platform="slack",
            channel_id="C2",
            user_id="U2",
        )

        # Cleanup
        count = await store.cleanup_expired()
        assert count == 1

        # Verify expired is gone
        expired = await store.get_origin("expired-1")
        assert expired is None

        # Valid still exists
        valid = await store.get_origin("valid-1")
        assert valid is not None

    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self, tmp_path):
        """Should evict oldest cache entries when full."""
        db_path = tmp_path / "cache_test.db"
        store = PersistentOriginStore(cache_size=3)
        store._sqlite_path = str(db_path)
        store._use_postgres = False
        await store.initialize()

        try:
            # Add 4 origins (cache size is 3)
            for i in range(4):
                await store.register_origin(
                    origin_id=f"origin-{i}",
                    origin_type="debate",
                    platform="slack",
                    channel_id=f"C{i}",
                    user_id=f"U{i}",
                )

            # First one should be evicted from cache
            assert "origin-0" not in store._cache
            assert "origin-1" in store._cache
            assert "origin-2" in store._cache
            assert "origin-3" in store._cache

            # But should still be retrievable from database
            origin = await store.get_origin("origin-0")
            assert origin is not None
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_persistence_across_instances(self, tmp_path):
        """Should persist data across store instances."""
        db_path = tmp_path / "persist_test.db"

        # Create and save with first instance
        store1 = PersistentOriginStore()
        store1._sqlite_path = str(db_path)
        store1._use_postgres = False
        await store1.initialize()

        await store1.register_origin(
            origin_id="persist-test",
            origin_type="debate",
            platform="slack",
            channel_id="C1",
            user_id="U1",
            metadata={"test": "data"},
        )

        await store1.close()

        # Create second instance and retrieve
        store2 = PersistentOriginStore()
        store2._sqlite_path = str(db_path)
        store2._use_postgres = False
        await store2.initialize()

        origin = await store2.get_origin("persist-test")
        assert origin is not None
        assert origin.origin_id == "persist-test"
        assert origin.metadata["test"] == "data"

        await store2.close()


class TestOriginStoreCaching:
    """Tests for cache behavior."""

    @pytest.fixture
    async def store(self, tmp_path):
        """Create store with small cache."""
        db_path = tmp_path / "cache_test.db"
        store = PersistentOriginStore(cache_size=5)
        store._sqlite_path = str(db_path)
        store._use_postgres = False
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_cache_hit(self, store):
        """Should return from cache on second access."""
        await store.register_origin(
            origin_id="cached-1",
            origin_type="debate",
            platform="slack",
            channel_id="C1",
            user_id="U1",
        )

        # First access (in cache from register)
        assert "cached-1" in store._cache

        # Get should return from cache
        origin = await store.get_origin("cached-1")
        assert origin is not None

    @pytest.mark.asyncio
    async def test_cache_miss_loads_from_db(self, store):
        """Should load from database on cache miss."""
        await store.register_origin(
            origin_id="db-load-1",
            origin_type="debate",
            platform="slack",
            channel_id="C1",
            user_id="U1",
        )

        # Clear cache
        store._cache.clear()
        store._cache_order.clear()

        # Should load from database
        origin = await store.get_origin("db-load-1")
        assert origin is not None
        assert "db-load-1" in store._cache
