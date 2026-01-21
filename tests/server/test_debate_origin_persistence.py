"""
Tests for debate origin SQLite persistence.

Tests cover:
- SQLiteOriginStore save/get operations
- Integration with register_debate_origin and get_debate_origin
- TTL cleanup of expired records
- Graceful fallback behavior
"""

import json
import os
import tempfile
import time
import pytest
from unittest.mock import patch, MagicMock

from aragora.server.debate_origin import (
    DebateOrigin,
    SQLiteOriginStore,
    register_debate_origin,
    get_debate_origin,
    mark_result_sent,
    cleanup_expired_origins,
    _origin_store,
    _get_sqlite_store,
    ORIGIN_TTL_SECONDS,
)


class TestSQLiteOriginStore:
    """Tests for SQLiteOriginStore class."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a store with temp database."""
        db_path = tmp_path / "test_origins.db"
        return SQLiteOriginStore(str(db_path))

    @pytest.fixture
    def sample_origin(self):
        """Sample debate origin."""
        return DebateOrigin(
            debate_id="debate-123",
            platform="slack",
            channel_id="C12345678",
            user_id="U87654321",
            thread_id="1234567890.123456",
            message_id="msg-001",
            metadata={"username": "testuser", "team_id": "T12345"},
        )

    def test_save_and_get(self, store, sample_origin):
        """Save and retrieve origin."""
        store.save(sample_origin)
        loaded = store.get(sample_origin.debate_id)

        assert loaded is not None
        assert loaded.debate_id == sample_origin.debate_id
        assert loaded.platform == sample_origin.platform
        assert loaded.channel_id == sample_origin.channel_id
        assert loaded.user_id == sample_origin.user_id
        assert loaded.thread_id == sample_origin.thread_id
        assert loaded.message_id == sample_origin.message_id
        assert loaded.metadata == sample_origin.metadata

    def test_get_nonexistent(self, store):
        """Get returns None for missing origin."""
        result = store.get("nonexistent-debate")
        assert result is None

    def test_update_existing(self, store, sample_origin):
        """Save updates existing origin."""
        store.save(sample_origin)

        # Update and save again
        sample_origin.result_sent = True
        sample_origin.result_sent_at = time.time()
        store.save(sample_origin)

        loaded = store.get(sample_origin.debate_id)
        assert loaded.result_sent is True
        assert loaded.result_sent_at is not None

    def test_metadata_serialization(self, store):
        """Complex metadata is properly serialized."""
        origin = DebateOrigin(
            debate_id="meta-test",
            platform="discord",
            channel_id="123456",
            user_id="789",
            metadata={
                "nested": {"key": "value"},
                "list": [1, 2, 3],
                "bool": True,
                "null": None,
            },
        )
        store.save(origin)
        loaded = store.get("meta-test")

        assert loaded.metadata["nested"] == {"key": "value"}
        assert loaded.metadata["list"] == [1, 2, 3]
        assert loaded.metadata["bool"] is True
        assert loaded.metadata["null"] is None

    def test_cleanup_expired(self, store):
        """cleanup_expired removes old records."""
        # Create an expired origin
        old_origin = DebateOrigin(
            debate_id="old-debate",
            platform="telegram",
            channel_id="111",
            user_id="222",
            created_at=time.time() - 100000,  # Way in the past
        )
        store.save(old_origin)

        # Create a fresh origin
        new_origin = DebateOrigin(
            debate_id="new-debate",
            platform="telegram",
            channel_id="333",
            user_id="444",
            created_at=time.time(),
        )
        store.save(new_origin)

        # Cleanup with short TTL
        count = store.cleanup_expired(ttl_seconds=1000)

        assert count == 1
        assert store.get("old-debate") is None
        assert store.get("new-debate") is not None


class TestRegisterDebateOrigin:
    """Tests for register_debate_origin with SQLite persistence."""

    @pytest.fixture(autouse=True)
    def clear_store(self):
        """Clear in-memory store before each test."""
        _origin_store.clear()
        yield
        _origin_store.clear()

    def test_persists_to_sqlite(self, tmp_path):
        """register_debate_origin saves to SQLite."""
        db_path = tmp_path / "test.db"
        mock_store = SQLiteOriginStore(str(db_path))

        with patch(
            "aragora.server.debate_origin._get_sqlite_store",
            return_value=mock_store
        ):
            origin = register_debate_origin(
                debate_id="persist-test",
                platform="slack",
                channel_id="C123",
                user_id="U456",
                metadata={"test": True},
            )

        # Verify persisted
        loaded = mock_store.get("persist-test")
        assert loaded is not None
        assert loaded.platform == "slack"
        assert loaded.metadata == {"test": True}

    def test_handles_sqlite_error(self):
        """register_debate_origin handles SQLite errors gracefully."""
        mock_store = MagicMock()
        mock_store.save.side_effect = Exception("DB error")

        with patch(
            "aragora.server.debate_origin._get_sqlite_store",
            return_value=mock_store
        ):
            # Should not raise
            origin = register_debate_origin(
                debate_id="error-test",
                platform="telegram",
                channel_id="123",
                user_id="456",
            )

        assert origin is not None
        assert origin.debate_id == "error-test"
        # Should still be in memory
        assert "error-test" in _origin_store


class TestGetDebateOrigin:
    """Tests for get_debate_origin with SQLite fallback."""

    @pytest.fixture(autouse=True)
    def clear_store(self):
        """Clear in-memory store before each test."""
        _origin_store.clear()
        yield
        _origin_store.clear()

    def test_returns_from_memory_first(self):
        """get_debate_origin returns in-memory origin first."""
        mem_origin = DebateOrigin(
            debate_id="mem-origin",
            platform="memory",
            channel_id="111",
            user_id="222",
        )
        _origin_store["mem-origin"] = mem_origin

        result = get_debate_origin("mem-origin")
        assert result is not None
        assert result.platform == "memory"

    def test_falls_back_to_sqlite(self, tmp_path):
        """get_debate_origin falls back to SQLite."""
        db_path = tmp_path / "fallback.db"
        mock_store = SQLiteOriginStore(str(db_path))

        # Save directly to SQLite (simulating restart)
        sqlite_origin = DebateOrigin(
            debate_id="sqlite-origin",
            platform="sqlite",
            channel_id="333",
            user_id="444",
        )
        mock_store.save(sqlite_origin)

        with patch(
            "aragora.server.debate_origin._get_sqlite_store",
            return_value=mock_store
        ):
            result = get_debate_origin("sqlite-origin")

        assert result is not None
        assert result.platform == "sqlite"
        # Should now be cached in memory
        assert "sqlite-origin" in _origin_store

    def test_returns_none_when_not_found(self):
        """get_debate_origin returns None when not found anywhere."""
        mock_store = MagicMock()
        mock_store.get.return_value = None

        with patch(
            "aragora.server.debate_origin._get_sqlite_store",
            return_value=mock_store
        ):
            result = get_debate_origin("nonexistent")

        assert result is None


class TestMarkResultSent:
    """Tests for mark_result_sent with SQLite persistence."""

    @pytest.fixture(autouse=True)
    def clear_store(self):
        """Clear in-memory store before each test."""
        _origin_store.clear()
        yield
        _origin_store.clear()

    def test_updates_sqlite(self, tmp_path):
        """mark_result_sent updates SQLite."""
        db_path = tmp_path / "mark.db"
        mock_store = SQLiteOriginStore(str(db_path))

        # Register origin
        origin = DebateOrigin(
            debate_id="mark-test",
            platform="slack",
            channel_id="C123",
            user_id="U456",
        )
        _origin_store["mark-test"] = origin
        mock_store.save(origin)

        with patch(
            "aragora.server.debate_origin._get_sqlite_store",
            return_value=mock_store
        ):
            mark_result_sent("mark-test")

        # Check SQLite was updated
        loaded = mock_store.get("mark-test")
        assert loaded.result_sent is True
        assert loaded.result_sent_at is not None


class TestDebateOriginDataclass:
    """Tests for DebateOrigin dataclass."""

    def test_to_dict(self):
        """to_dict serializes all fields."""
        origin = DebateOrigin(
            debate_id="dict-test",
            platform="telegram",
            channel_id="123",
            user_id="456",
            thread_id="789",
            message_id="msg-1",
            metadata={"key": "value"},
            result_sent=True,
            result_sent_at=1234567890.0,
        )

        d = origin.to_dict()

        assert d["debate_id"] == "dict-test"
        assert d["platform"] == "telegram"
        assert d["metadata"] == {"key": "value"}
        assert d["result_sent"] is True
        assert d["result_sent_at"] == 1234567890.0

    def test_from_dict(self):
        """from_dict deserializes correctly."""
        data = {
            "debate_id": "from-dict",
            "platform": "discord",
            "channel_id": "111",
            "user_id": "222",
            "created_at": 1234567890.0,
            "metadata": {"foo": "bar"},
            "thread_id": "t1",
            "message_id": "m1",
            "result_sent": True,
            "result_sent_at": 1234567899.0,
        }

        origin = DebateOrigin.from_dict(data)

        assert origin.debate_id == "from-dict"
        assert origin.platform == "discord"
        assert origin.created_at == 1234567890.0
        assert origin.metadata == {"foo": "bar"}
        assert origin.result_sent is True
