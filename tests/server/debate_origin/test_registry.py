"""Comprehensive tests for debate origin registry.

Tests cover:
1. Origin registration and lookup (Redis + in-memory fallback)
2. TTL expiry of origins
3. Concurrent registrations
4. Redis storage and loading
5. Error handling and fallback behavior
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from aragora.server.debate_origin import (
    DebateOrigin,
    ORIGIN_TTL_SECONDS,
    _origin_store,
    register_debate_origin,
    get_debate_origin,
    get_debate_origin_async,
    mark_result_sent,
    cleanup_expired_origins,
)
from aragora.server.debate_origin.registry import (
    _store_origin_redis,
    _load_origin_redis,
    _resolve_store_origin_redis,
    _resolve_load_origin_redis,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_origin_store():
    """Clear in-memory store before and after each test."""
    _origin_store.clear()
    yield
    _origin_store.clear()


@pytest.fixture
def mock_sqlite_store():
    """Create a mock SQLite store."""
    store = MagicMock()
    store.save = MagicMock()
    store.save_async = AsyncMock()
    store.get = MagicMock(return_value=None)
    store.get_async = AsyncMock(return_value=None)
    store.cleanup_expired = MagicMock(return_value=0)
    return store


@pytest.fixture
def mock_redis():
    """Mock Redis module and client."""
    mock_client = MagicMock()
    mock_client.setex = MagicMock()
    mock_client.get = MagicMock(return_value=None)

    with patch("redis.from_url", return_value=mock_client) as mock_from_url:
        yield {"client": mock_client, "from_url": mock_from_url}


@pytest.fixture
def sample_origin():
    """Create a sample DebateOrigin."""
    return DebateOrigin(
        debate_id="test-debate-123",
        platform="telegram",
        channel_id="123456789",
        user_id="987654321",
        thread_id="thread-1",
        message_id="msg-1",
        metadata={"username": "testuser"},
    )


# =============================================================================
# Test: Origin Registration
# =============================================================================


class TestRegisterDebateOrigin:
    """Tests for register_debate_origin function."""

    def test_registers_origin_in_memory(self, mock_sqlite_store):
        """register_debate_origin stores origin in memory."""
        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._resolve_store_origin_redis",
                return_value=MagicMock(side_effect=ImportError),
            ):
                with patch(
                    "aragora.server.debate_origin.registry._get_postgres_store_sync",
                    return_value=None,
                ):
                    origin = register_debate_origin(
                        debate_id="mem-test-001",
                        platform="slack",
                        channel_id="C12345",
                        user_id="U67890",
                    )

        assert "mem-test-001" in _origin_store
        assert origin.platform == "slack"
        assert origin.channel_id == "C12345"

    def test_registers_with_all_fields(self, mock_sqlite_store):
        """register_debate_origin handles all optional fields."""
        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._resolve_store_origin_redis",
                return_value=MagicMock(side_effect=ImportError),
            ):
                with patch(
                    "aragora.server.debate_origin.registry._get_postgres_store_sync",
                    return_value=None,
                ):
                    origin = register_debate_origin(
                        debate_id="full-test-001",
                        platform="discord",
                        channel_id="123456789012345678",
                        user_id="876543210987654321",
                        thread_id="thread-xyz",
                        message_id="msg-abc",
                        metadata={"guild_id": "111222333", "channel_name": "general"},
                        session_id="session-123",
                    )

        assert origin.thread_id == "thread-xyz"
        assert origin.message_id == "msg-abc"
        assert origin.metadata["guild_id"] == "111222333"
        assert origin.session_id == "session-123"

    def test_persists_to_sqlite(self, mock_sqlite_store):
        """register_debate_origin persists to SQLite store."""
        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._resolve_store_origin_redis",
                return_value=MagicMock(side_effect=ImportError),
            ):
                with patch(
                    "aragora.server.debate_origin.registry._get_postgres_store_sync",
                    return_value=None,
                ):
                    origin = register_debate_origin(
                        debate_id="sqlite-test",
                        platform="telegram",
                        channel_id="111",
                        user_id="222",
                    )

        mock_sqlite_store.save.assert_called_once()
        saved_origin = mock_sqlite_store.save.call_args[0][0]
        assert saved_origin.debate_id == "sqlite-test"

    def test_handles_sqlite_error_gracefully(self, mock_sqlite_store):
        """register_debate_origin handles SQLite errors gracefully."""
        import sqlite3

        mock_sqlite_store.save.side_effect = sqlite3.OperationalError("DB locked")

        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._resolve_store_origin_redis",
                return_value=MagicMock(side_effect=ImportError),
            ):
                with patch(
                    "aragora.server.debate_origin.registry._get_postgres_store_sync",
                    return_value=None,
                ):
                    # Should not raise - the function catches sqlite3.OperationalError
                    origin = register_debate_origin(
                        debate_id="error-test",
                        platform="slack",
                        channel_id="C1",
                        user_id="U1",
                    )

        # Origin should still be in memory
        assert origin is not None
        assert origin.debate_id == "error-test"
        assert "error-test" in _origin_store

    def test_persists_to_redis_when_available(self, mock_sqlite_store):
        """register_debate_origin persists to Redis when available."""
        mock_redis_fn = MagicMock()

        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._resolve_store_origin_redis",
                return_value=mock_redis_fn,
            ):
                with patch(
                    "aragora.server.debate_origin.registry._get_postgres_store_sync",
                    return_value=None,
                ):
                    origin = register_debate_origin(
                        debate_id="redis-test",
                        platform="teams",
                        channel_id="T123",
                        user_id="U456",
                    )

        mock_redis_fn.assert_called_once()
        stored_origin = mock_redis_fn.call_args[0][0]
        assert stored_origin.debate_id == "redis-test"

    def test_handles_redis_connection_error(self, mock_sqlite_store):
        """register_debate_origin handles Redis connection errors."""
        mock_redis_fn = MagicMock(side_effect=ConnectionError("Redis unavailable"))

        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._resolve_store_origin_redis",
                return_value=mock_redis_fn,
            ):
                with patch(
                    "aragora.server.debate_origin.registry._get_postgres_store_sync",
                    return_value=None,
                ):
                    # Should not raise - fallback to SQLite
                    origin = register_debate_origin(
                        debate_id="redis-error-test",
                        platform="whatsapp",
                        channel_id="+1234567890",
                        user_id="user123",
                    )

        assert origin.debate_id == "redis-error-test"
        assert "redis-error-test" in _origin_store

    def test_sets_created_at_timestamp(self, mock_sqlite_store):
        """register_debate_origin sets created_at timestamp."""
        before = time.time()

        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._resolve_store_origin_redis",
                return_value=MagicMock(side_effect=ImportError),
            ):
                with patch(
                    "aragora.server.debate_origin.registry._get_postgres_store_sync",
                    return_value=None,
                ):
                    origin = register_debate_origin(
                        debate_id="timestamp-test",
                        platform="email",
                        channel_id="inbox",
                        user_id="user@example.com",
                    )

        after = time.time()
        assert before <= origin.created_at <= after


# =============================================================================
# Test: Origin Lookup
# =============================================================================


class TestGetDebateOrigin:
    """Tests for get_debate_origin function."""

    def test_returns_from_memory(self, sample_origin):
        """get_debate_origin returns origin from memory first."""
        _origin_store[sample_origin.debate_id] = sample_origin

        result = get_debate_origin(sample_origin.debate_id)

        assert result is sample_origin

    def test_returns_none_for_missing(self, mock_sqlite_store):
        """get_debate_origin returns None when not found."""
        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._resolve_load_origin_redis",
                return_value=MagicMock(side_effect=ImportError),
            ):
                with patch(
                    "aragora.server.debate_origin.registry._get_postgres_store_sync",
                    return_value=None,
                ):
                    result = get_debate_origin("nonexistent-id")

        assert result is None

    def test_falls_back_to_redis(self, sample_origin, mock_sqlite_store):
        """get_debate_origin falls back to Redis when not in memory."""
        mock_redis_fn = MagicMock(return_value=sample_origin)

        with patch(
            "aragora.server.debate_origin.registry._resolve_load_origin_redis",
            return_value=mock_redis_fn,
        ):
            result = get_debate_origin(sample_origin.debate_id)

        assert result == sample_origin
        # Should be cached in memory
        assert sample_origin.debate_id in _origin_store

    def test_falls_back_to_sqlite(self, sample_origin, mock_sqlite_store):
        """get_debate_origin falls back to SQLite when Redis unavailable."""
        mock_sqlite_store.get.return_value = sample_origin

        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._resolve_load_origin_redis",
                return_value=MagicMock(side_effect=ImportError),
            ):
                with patch(
                    "aragora.server.debate_origin.registry._get_postgres_store_sync",
                    return_value=None,
                ):
                    result = get_debate_origin(sample_origin.debate_id)

        assert result == sample_origin
        mock_sqlite_store.get.assert_called_once_with(sample_origin.debate_id)

    def test_caches_result_in_memory(self, sample_origin, mock_sqlite_store):
        """get_debate_origin caches loaded origin in memory."""
        mock_sqlite_store.get.return_value = sample_origin

        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._resolve_load_origin_redis",
                return_value=MagicMock(side_effect=ImportError),
            ):
                with patch(
                    "aragora.server.debate_origin.registry._get_postgres_store_sync",
                    return_value=None,
                ):
                    get_debate_origin(sample_origin.debate_id)

        # Second call should use memory
        result = get_debate_origin(sample_origin.debate_id)
        assert result == sample_origin

    def test_handles_redis_json_decode_error(self, mock_sqlite_store):
        """get_debate_origin handles Redis JSON decode errors."""
        mock_redis_fn = MagicMock(side_effect=json.JSONDecodeError("error", "", 0))
        mock_sqlite_store.get.return_value = None

        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._resolve_load_origin_redis",
                return_value=mock_redis_fn,
            ):
                with patch(
                    "aragora.server.debate_origin.registry._get_postgres_store_sync",
                    return_value=None,
                ):
                    result = get_debate_origin("corrupt-data")

        assert result is None


# =============================================================================
# Test: Async Origin Lookup
# =============================================================================


class TestGetDebateOriginAsync:
    """Tests for get_debate_origin_async function."""

    @pytest.mark.asyncio
    async def test_returns_from_memory(self, sample_origin):
        """get_debate_origin_async returns origin from memory."""
        _origin_store[sample_origin.debate_id] = sample_origin

        result = await get_debate_origin_async(sample_origin.debate_id)

        assert result is sample_origin

    @pytest.mark.asyncio
    async def test_falls_back_to_sqlite_async(self, sample_origin, mock_sqlite_store):
        """get_debate_origin_async uses async SQLite method."""
        mock_sqlite_store.get_async.return_value = sample_origin

        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._resolve_load_origin_redis",
                return_value=MagicMock(side_effect=ImportError),
            ):
                with patch(
                    "aragora.server.debate_origin.registry._get_postgres_store",
                    new_callable=AsyncMock,
                    return_value=None,
                ):
                    result = await get_debate_origin_async(sample_origin.debate_id)

        assert result == sample_origin
        mock_sqlite_store.get_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, mock_sqlite_store):
        """get_debate_origin_async returns None when not found anywhere."""
        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._resolve_load_origin_redis",
                return_value=MagicMock(side_effect=ImportError),
            ):
                with patch(
                    "aragora.server.debate_origin.registry._get_postgres_store",
                    new_callable=AsyncMock,
                    return_value=None,
                ):
                    result = await get_debate_origin_async("missing-123")

        assert result is None


# =============================================================================
# Test: Mark Result Sent
# =============================================================================


class TestMarkResultSent:
    """Tests for mark_result_sent function."""

    def test_updates_result_sent_flag(self, sample_origin, mock_sqlite_store):
        """mark_result_sent sets result_sent to True."""
        _origin_store[sample_origin.debate_id] = sample_origin

        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._resolve_store_origin_redis",
                return_value=MagicMock(side_effect=ImportError),
            ):
                with patch(
                    "aragora.server.debate_origin.registry._get_postgres_store_sync",
                    return_value=None,
                ):
                    mark_result_sent(sample_origin.debate_id)

        assert sample_origin.result_sent is True

    def test_sets_result_sent_at_timestamp(self, sample_origin, mock_sqlite_store):
        """mark_result_sent sets result_sent_at timestamp."""
        _origin_store[sample_origin.debate_id] = sample_origin
        before = time.time()

        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._resolve_store_origin_redis",
                return_value=MagicMock(side_effect=ImportError),
            ):
                with patch(
                    "aragora.server.debate_origin.registry._get_postgres_store_sync",
                    return_value=None,
                ):
                    mark_result_sent(sample_origin.debate_id)

        after = time.time()
        assert sample_origin.result_sent_at is not None
        assert before <= sample_origin.result_sent_at <= after

    def test_updates_sqlite(self, sample_origin, mock_sqlite_store):
        """mark_result_sent updates SQLite store."""
        _origin_store[sample_origin.debate_id] = sample_origin

        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._resolve_store_origin_redis",
                return_value=MagicMock(side_effect=ImportError),
            ):
                with patch(
                    "aragora.server.debate_origin.registry._get_postgres_store_sync",
                    return_value=None,
                ):
                    mark_result_sent(sample_origin.debate_id)

        mock_sqlite_store.save.assert_called()

    def test_updates_redis(self, sample_origin, mock_sqlite_store):
        """mark_result_sent updates Redis."""
        _origin_store[sample_origin.debate_id] = sample_origin
        mock_redis_fn = MagicMock()

        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._resolve_store_origin_redis",
                return_value=mock_redis_fn,
            ):
                with patch(
                    "aragora.server.debate_origin.registry._get_postgres_store_sync",
                    return_value=None,
                ):
                    mark_result_sent(sample_origin.debate_id)

        mock_redis_fn.assert_called()

    def test_handles_missing_origin(self, mock_sqlite_store):
        """mark_result_sent handles missing origin gracefully."""
        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._resolve_load_origin_redis",
                return_value=MagicMock(side_effect=ImportError),
            ):
                with patch(
                    "aragora.server.debate_origin.registry._get_postgres_store_sync",
                    return_value=None,
                ):
                    # Should not raise
                    mark_result_sent("nonexistent-origin")


# =============================================================================
# Test: TTL Expiry and Cleanup
# =============================================================================


class TestCleanupExpiredOrigins:
    """Tests for cleanup_expired_origins function."""

    def test_removes_expired_from_memory(self, mock_sqlite_store):
        """cleanup_expired_origins removes expired origins from memory."""
        # Add expired origin
        expired = DebateOrigin(
            debate_id="expired-001",
            platform="telegram",
            channel_id="111",
            user_id="222",
            created_at=time.time() - ORIGIN_TTL_SECONDS - 3600,  # 1 hour past expiry
        )
        _origin_store["expired-001"] = expired

        # Add fresh origin
        fresh = DebateOrigin(
            debate_id="fresh-001",
            platform="telegram",
            channel_id="333",
            user_id="444",
            created_at=time.time(),
        )
        _origin_store["fresh-001"] = fresh

        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._get_postgres_store_sync",
                return_value=None,
            ):
                count = cleanup_expired_origins()

        assert count >= 1
        assert "expired-001" not in _origin_store
        assert "fresh-001" in _origin_store

    def test_cleans_up_sqlite_store(self, mock_sqlite_store):
        """cleanup_expired_origins cleans up SQLite store."""
        mock_sqlite_store.cleanup_expired.return_value = 5

        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._get_postgres_store_sync",
                return_value=None,
            ):
                count = cleanup_expired_origins()

        mock_sqlite_store.cleanup_expired.assert_called_once_with(ORIGIN_TTL_SECONDS)
        assert count >= 5

    def test_returns_total_cleaned_count(self, mock_sqlite_store):
        """cleanup_expired_origins returns total count from all sources."""
        # Add one expired origin to memory
        expired = DebateOrigin(
            debate_id="expired-mem",
            platform="slack",
            channel_id="C1",
            user_id="U1",
            created_at=time.time() - ORIGIN_TTL_SECONDS - 1000,
        )
        _origin_store["expired-mem"] = expired

        # SQLite reports 3 cleaned
        mock_sqlite_store.cleanup_expired.return_value = 3

        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._get_postgres_store_sync",
                return_value=None,
            ):
                count = cleanup_expired_origins()

        # 1 from memory + 3 from SQLite
        assert count == 4

    def test_handles_empty_store(self, mock_sqlite_store):
        """cleanup_expired_origins handles empty stores."""
        mock_sqlite_store.cleanup_expired.return_value = 0

        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._get_postgres_store_sync",
                return_value=None,
            ):
                count = cleanup_expired_origins()

        assert count == 0


# =============================================================================
# Test: Redis Storage Functions
# =============================================================================


class TestRedisStorage:
    """Tests for Redis storage functions."""

    def test_store_origin_redis_sets_with_ttl(self, sample_origin):
        """_store_origin_redis sets key with TTL."""
        mock_redis_client = MagicMock()

        with patch("redis.from_url", return_value=mock_redis_client):
            _store_origin_redis(sample_origin)

        mock_redis_client.setex.assert_called_once()
        call_args = mock_redis_client.setex.call_args
        assert call_args[0][0] == f"debate_origin:{sample_origin.debate_id}"
        assert call_args[0][1] == ORIGIN_TTL_SECONDS

    def test_store_origin_redis_serializes_correctly(self, sample_origin):
        """_store_origin_redis serializes origin to JSON."""
        mock_redis_client = MagicMock()

        with patch("redis.from_url", return_value=mock_redis_client):
            _store_origin_redis(sample_origin)

        stored_json = mock_redis_client.setex.call_args[0][2]
        data = json.loads(stored_json)
        assert data["debate_id"] == sample_origin.debate_id
        assert data["platform"] == sample_origin.platform

    def test_load_origin_redis_returns_origin(self, sample_origin):
        """_load_origin_redis returns DebateOrigin from Redis."""
        mock_redis_client = MagicMock()
        mock_redis_client.get.return_value = json.dumps(sample_origin.to_dict())

        with patch("redis.from_url", return_value=mock_redis_client):
            result = _load_origin_redis(sample_origin.debate_id)

        assert result.debate_id == sample_origin.debate_id
        assert result.platform == sample_origin.platform

    def test_load_origin_redis_returns_none_when_not_found(self):
        """_load_origin_redis returns None when key doesn't exist."""
        mock_redis_client = MagicMock()
        mock_redis_client.get.return_value = None

        with patch("redis.from_url", return_value=mock_redis_client):
            result = _load_origin_redis("nonexistent")

        assert result is None

    def test_store_origin_redis_raises_on_import_error(self, sample_origin):
        """_store_origin_redis raises ImportError when redis not installed."""
        with patch.dict("sys.modules", {"redis": None}):
            with pytest.raises(ImportError):
                _store_origin_redis(sample_origin)


# =============================================================================
# Test: Concurrent Registrations
# =============================================================================


class TestConcurrentRegistrations:
    """Tests for concurrent origin registration."""

    @pytest.mark.asyncio
    async def test_concurrent_registrations_no_data_loss(self, mock_sqlite_store):
        """Multiple concurrent registrations don't lose data."""
        ids = [f"concurrent-{i}" for i in range(10)]

        async def register_one(debate_id: str):
            with patch(
                "aragora.server.debate_origin.registry._get_sqlite_store",
                return_value=mock_sqlite_store,
            ):
                with patch(
                    "aragora.server.debate_origin.registry._resolve_store_origin_redis",
                    return_value=MagicMock(side_effect=ImportError),
                ):
                    with patch(
                        "aragora.server.debate_origin.registry._get_postgres_store_sync",
                        return_value=None,
                    ):
                        return register_debate_origin(
                            debate_id=debate_id,
                            platform="telegram",
                            channel_id=f"ch-{debate_id}",
                            user_id=f"user-{debate_id}",
                        )

        # Register all concurrently
        results = await asyncio.gather(*[register_one(id) for id in ids])

        # All should be in memory
        assert len(results) == 10
        for id in ids:
            assert id in _origin_store

    @pytest.mark.asyncio
    async def test_concurrent_lookups_with_cache(self, sample_origin, mock_sqlite_store):
        """Concurrent lookups properly cache result."""
        mock_sqlite_store.get.return_value = sample_origin

        async def lookup_one():
            with patch(
                "aragora.server.debate_origin.registry._get_sqlite_store",
                return_value=mock_sqlite_store,
            ):
                with patch(
                    "aragora.server.debate_origin.registry._resolve_load_origin_redis",
                    return_value=MagicMock(side_effect=ImportError),
                ):
                    with patch(
                        "aragora.server.debate_origin.registry._get_postgres_store_sync",
                        return_value=None,
                    ):
                        return get_debate_origin(sample_origin.debate_id)

        # Look up concurrently
        results = await asyncio.gather(*[lookup_one() for _ in range(5)])

        # All should return the same origin
        assert all(r.debate_id == sample_origin.debate_id for r in results)


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_metadata(self, mock_sqlite_store):
        """Registration works with empty metadata."""
        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._resolve_store_origin_redis",
                return_value=MagicMock(side_effect=ImportError),
            ):
                with patch(
                    "aragora.server.debate_origin.registry._get_postgres_store_sync",
                    return_value=None,
                ):
                    origin = register_debate_origin(
                        debate_id="no-meta",
                        platform="slack",
                        channel_id="C1",
                        user_id="U1",
                        metadata=None,
                    )

        assert origin.metadata == {}

    def test_special_characters_in_ids(self, mock_sqlite_store):
        """Registration handles special characters in IDs."""
        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._resolve_store_origin_redis",
                return_value=MagicMock(side_effect=ImportError),
            ):
                with patch(
                    "aragora.server.debate_origin.registry._get_postgres_store_sync",
                    return_value=None,
                ):
                    origin = register_debate_origin(
                        debate_id="debate:with:colons:and/slashes",
                        platform="telegram",
                        channel_id="-1001234567890",  # Telegram supergroup ID
                        user_id="user@example.com",
                    )

        assert origin.debate_id == "debate:with:colons:and/slashes"
        assert "debate:with:colons:and/slashes" in _origin_store

    def test_unicode_in_metadata(self, mock_sqlite_store):
        """Registration handles unicode in metadata."""
        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._resolve_store_origin_redis",
                return_value=MagicMock(side_effect=ImportError),
            ):
                with patch(
                    "aragora.server.debate_origin.registry._get_postgres_store_sync",
                    return_value=None,
                ):
                    origin = register_debate_origin(
                        debate_id="unicode-test",
                        platform="telegram",
                        channel_id="123",
                        user_id="456",
                        metadata={"username": "TestUser", "language": "ru"},
                    )

        assert origin.metadata["username"] == "TestUser"
        assert origin.metadata["language"] == "ru"

    def test_very_long_debate_id(self, mock_sqlite_store):
        """Registration handles very long debate IDs."""
        long_id = "d" * 500

        with patch(
            "aragora.server.debate_origin.registry._get_sqlite_store",
            return_value=mock_sqlite_store,
        ):
            with patch(
                "aragora.server.debate_origin.registry._resolve_store_origin_redis",
                return_value=MagicMock(side_effect=ImportError),
            ):
                with patch(
                    "aragora.server.debate_origin.registry._get_postgres_store_sync",
                    return_value=None,
                ):
                    origin = register_debate_origin(
                        debate_id=long_id,
                        platform="slack",
                        channel_id="C1",
                        user_id="U1",
                    )

        assert origin.debate_id == long_id
        assert long_id in _origin_store
