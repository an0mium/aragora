"""
Tests for GmailTokenStore backends.

Tests all three backends: InMemory, SQLite, and Redis (with fallback).
"""

import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aragora.storage.gmail_token_store import (
    GmailUserState,
    InMemoryGmailTokenStore,
    RedisGmailTokenStore,
    SQLiteGmailTokenStore,
    SyncJobState,
    get_gmail_token_store,
    reset_gmail_token_store,
    set_gmail_token_store,
)


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_gmail_tokens.db"


@pytest.fixture
def memory_store():
    """Create an in-memory store for testing."""
    return InMemoryGmailTokenStore()


@pytest.fixture
def sqlite_store(temp_db_path):
    """Create a SQLite store for testing."""
    return SQLiteGmailTokenStore(temp_db_path)


@pytest.fixture
def sample_state():
    """Create a sample Gmail user state."""
    return GmailUserState(
        user_id="user123",
        email_address="user@example.com",
        access_token="access_token_123",
        refresh_token="refresh_token_456",
        token_expiry=datetime.now(timezone.utc),
        history_id="12345",
        indexed_count=100,
        connected_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_job():
    """Create a sample sync job state."""
    return SyncJobState(
        user_id="user123",
        status="running",
        progress=50,
        messages_synced=25,
        started_at=datetime.now(timezone.utc).isoformat(),
    )


class TestGmailUserState:
    """Tests for GmailUserState dataclass."""

    def test_to_dict_masks_tokens(self, sample_state):
        """Tokens should not be included by default."""
        result = sample_state.to_dict(include_tokens=False)
        assert "access_token" not in result
        assert "refresh_token" not in result
        assert result["email_address"] == "user@example.com"
        assert result["is_connected"] is True

    def test_to_dict_includes_tokens(self, sample_state):
        """Tokens should be included when requested."""
        result = sample_state.to_dict(include_tokens=True)
        assert result["access_token"] == "access_token_123"
        assert result["refresh_token"] == "refresh_token_456"

    def test_to_json_roundtrip(self, sample_state):
        """JSON serialization should preserve data."""
        json_str = sample_state.to_json()
        restored = GmailUserState.from_json(json_str)
        assert restored.user_id == sample_state.user_id
        assert restored.email_address == sample_state.email_address
        assert restored.access_token == sample_state.access_token
        assert restored.refresh_token == sample_state.refresh_token
        assert restored.indexed_count == sample_state.indexed_count

    def test_is_connected(self, sample_state):
        """is_connected should reflect refresh_token presence."""
        assert sample_state.to_dict()["is_connected"] is True
        sample_state.refresh_token = ""
        assert sample_state.to_dict()["is_connected"] is False


class TestSyncJobState:
    """Tests for SyncJobState dataclass."""

    def test_to_dict(self, sample_job):
        """Should serialize to dict."""
        result = sample_job.to_dict()
        assert result["user_id"] == "user123"
        assert result["status"] == "running"
        assert result["progress"] == 50
        assert result["messages_synced"] == 25


class TestInMemoryGmailTokenStore:
    """Tests for InMemoryGmailTokenStore."""

    @pytest.mark.asyncio
    async def test_save_and_get(self, memory_store, sample_state):
        """Should save and retrieve state."""
        await memory_store.save(sample_state)
        retrieved = await memory_store.get("user123")
        assert retrieved is not None
        assert retrieved.email_address == "user@example.com"
        assert retrieved.access_token == "access_token_123"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, memory_store):
        """Should return None for nonexistent user."""
        result = await memory_store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, memory_store, sample_state):
        """Should delete state."""
        await memory_store.save(sample_state)
        deleted = await memory_store.delete("user123")
        assert deleted is True
        result = await memory_store.get("user123")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, memory_store):
        """Should return False when deleting nonexistent."""
        deleted = await memory_store.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_all(self, memory_store):
        """Should list all states."""
        states = [
            GmailUserState(user_id="user1", email_address="user1@example.com"),
            GmailUserState(user_id="user2", email_address="user2@example.com"),
        ]
        for state in states:
            await memory_store.save(state)

        all_states = await memory_store.list_all()
        assert len(all_states) == 2

    @pytest.mark.asyncio
    async def test_sync_job_save_and_get(self, memory_store, sample_job):
        """Should save and retrieve sync job."""
        await memory_store.save_sync_job(sample_job)
        retrieved = await memory_store.get_sync_job("user123")
        assert retrieved is not None
        assert retrieved.status == "running"
        assert retrieved.progress == 50

    @pytest.mark.asyncio
    async def test_sync_job_delete(self, memory_store, sample_job):
        """Should delete sync job."""
        await memory_store.save_sync_job(sample_job)
        deleted = await memory_store.delete_sync_job("user123")
        assert deleted is True
        result = await memory_store.get_sync_job("user123")
        assert result is None


class TestSQLiteGmailTokenStore:
    """Tests for SQLiteGmailTokenStore."""

    @pytest.mark.asyncio
    async def test_save_and_get(self, sqlite_store, sample_state):
        """Should save and retrieve state."""
        await sqlite_store.save(sample_state)
        retrieved = await sqlite_store.get("user123")
        assert retrieved is not None
        assert retrieved.email_address == "user@example.com"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, sqlite_store):
        """Should return None for nonexistent user."""
        result = await sqlite_store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, sqlite_store, sample_state):
        """Should delete state."""
        await sqlite_store.save(sample_state)
        deleted = await sqlite_store.delete("user123")
        assert deleted is True
        result = await sqlite_store.get("user123")
        assert result is None

    @pytest.mark.asyncio
    async def test_persistence(self, temp_db_path, sample_state):
        """Data should persist across store instances."""
        store1 = SQLiteGmailTokenStore(temp_db_path)
        await store1.save(sample_state)
        await store1.close()

        store2 = SQLiteGmailTokenStore(temp_db_path)
        retrieved = await store2.get("user123")
        assert retrieved is not None
        assert retrieved.email_address == "user@example.com"
        await store2.close()

    @pytest.mark.asyncio
    async def test_sync_job_persistence(self, temp_db_path, sample_job):
        """Sync jobs should persist."""
        store1 = SQLiteGmailTokenStore(temp_db_path)
        await store1.save_sync_job(sample_job)
        await store1.close()

        store2 = SQLiteGmailTokenStore(temp_db_path)
        retrieved = await store2.get_sync_job("user123")
        assert retrieved is not None
        assert retrieved.status == "running"
        await store2.close()

    @pytest.mark.asyncio
    async def test_update_existing(self, sqlite_store, sample_state):
        """Should update existing state."""
        await sqlite_store.save(sample_state)
        sample_state.indexed_count = 200
        sample_state.email_address = "updated@example.com"
        await sqlite_store.save(sample_state)

        retrieved = await sqlite_store.get("user123")
        assert retrieved is not None
        assert retrieved.indexed_count == 200
        assert retrieved.email_address == "updated@example.com"


class TestRedisGmailTokenStore:
    """Tests for RedisGmailTokenStore (with SQLite fallback)."""

    @pytest.fixture
    def redis_store(self, temp_db_path):
        """Create a Redis store (will use SQLite fallback if Redis unavailable)."""
        return RedisGmailTokenStore(temp_db_path, redis_url="redis://localhost:6379")

    @pytest.mark.asyncio
    async def test_save_and_get(self, redis_store, sample_state):
        """Should save and retrieve state."""
        await redis_store.save(sample_state)
        retrieved = await redis_store.get("user123")
        assert retrieved is not None
        assert retrieved.email_address == "user@example.com"

    @pytest.mark.asyncio
    async def test_delete(self, redis_store, sample_state):
        """Should delete state."""
        await redis_store.save(sample_state)
        deleted = await redis_store.delete("user123")
        assert deleted is True

    @pytest.mark.asyncio
    async def test_fallback_persistence(self, temp_db_path, sample_state):
        """SQLite fallback should persist data."""
        store1 = RedisGmailTokenStore(temp_db_path)
        await store1.save(sample_state)
        await store1.close()

        store2 = RedisGmailTokenStore(temp_db_path)
        retrieved = await store2.get("user123")
        assert retrieved is not None
        await store2.close()


class TestGlobalStore:
    """Tests for global store factory functions."""

    def setup_method(self):
        """Reset global store before each test."""
        reset_gmail_token_store()

    def teardown_method(self):
        """Reset global store after each test."""
        reset_gmail_token_store()

    def test_get_default_store(self, monkeypatch, temp_db_path):
        """Should create default SQLite store."""
        monkeypatch.setenv("ARAGORA_DATA_DIR", str(temp_db_path.parent))
        store = get_gmail_token_store()
        assert isinstance(store, SQLiteGmailTokenStore)

    def test_get_memory_store(self, monkeypatch):
        """Should create in-memory store when configured."""
        monkeypatch.setenv("ARAGORA_GMAIL_STORE_BACKEND", "memory")
        store = get_gmail_token_store()
        assert isinstance(store, InMemoryGmailTokenStore)

    def test_set_custom_store(self):
        """Should allow setting custom store."""
        custom_store = InMemoryGmailTokenStore()
        set_gmail_token_store(custom_store)
        store = get_gmail_token_store()
        assert store is custom_store

    def test_singleton_pattern(self, monkeypatch):
        """Should return same instance on multiple calls."""
        monkeypatch.setenv("ARAGORA_GMAIL_STORE_BACKEND", "memory")
        store1 = get_gmail_token_store()
        store2 = get_gmail_token_store()
        assert store1 is store2


class TestGmailUserStateFromRow:
    """Tests for GmailUserState.from_row database deserialization."""

    def test_from_row_basic(self):
        """Should deserialize from database row."""
        row = (
            "user123",  # user_id
            "user@example.com",  # email_address
            "access_token",  # access_token
            "refresh_token",  # refresh_token
            "2024-01-01T00:00:00+00:00",  # token_expiry
            "12345",  # history_id
            "2024-01-01T00:00:00+00:00",  # last_sync
            100,  # indexed_count
            500,  # total_count
            "2024-01-01T00:00:00+00:00",  # connected_at
            1700000000.0,  # created_at
            1700000001.0,  # updated_at
        )
        state = GmailUserState.from_row(row)
        assert state.user_id == "user123"
        assert state.email_address == "user@example.com"
        assert state.access_token == "access_token"
        assert state.indexed_count == 100

    def test_from_row_null_values(self):
        """Should handle null values."""
        row = ("user123", None, None, None, None, None, None, None, None, None, None, None)
        state = GmailUserState.from_row(row)
        assert state.user_id == "user123"
        assert state.email_address == ""
        assert state.token_expiry is None
        assert state.indexed_count == 0
