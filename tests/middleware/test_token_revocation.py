"""
Tests for Token Revocation Middleware.

Tests cover:
- RevocationEntry dataclass
- InMemoryRevocationStore operations
- RedisRevocationStore operations (mocked)
- Token revocation lifecycle
- Blacklist checking
- TTL-based expiration and cleanup
- Thread safety
- Error handling and edge cases
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from aragora.server.middleware.token_revocation import (
    RevocationEntry,
    InMemoryRevocationStore,
    RedisRevocationStore,
    get_revocation_store,
    hash_token,
    revoke_token,
    is_token_revoked,
    unrevoke_token,
    get_revocation_stats,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_global_store():
    """Reset global revocation store before each test."""
    import aragora.server.middleware.token_revocation as module

    module._revocation_store = None
    yield
    module._revocation_store = None


@pytest.fixture
def memory_store():
    """Create a fresh in-memory revocation store."""
    return InMemoryRevocationStore(cleanup_interval=300.0)


@pytest.fixture
def sample_entry():
    """Create a sample revocation entry."""
    now = datetime.now(timezone.utc)
    return RevocationEntry(
        token_hash="abc123def456",
        revoked_at=now,
        expires_at=now + timedelta(hours=24),
        reason="logout",
        revoked_by="user-123",
        metadata={"session_id": "sess-456"},
    )


# ============================================================================
# RevocationEntry Tests
# ============================================================================


class TestRevocationEntry:
    """Tests for RevocationEntry dataclass."""

    def test_entry_creation(self, sample_entry):
        """Test creating a revocation entry with all fields."""
        assert sample_entry.token_hash == "abc123def456"
        assert sample_entry.reason == "logout"
        assert sample_entry.revoked_by == "user-123"
        assert sample_entry.metadata == {"session_id": "sess-456"}

    def test_entry_defaults(self):
        """Test RevocationEntry default values."""
        now = datetime.now(timezone.utc)
        entry = RevocationEntry(
            token_hash="hash123",
            revoked_at=now,
            expires_at=now + timedelta(hours=1),
        )

        assert entry.reason == ""
        assert entry.revoked_by == ""
        assert entry.metadata == {}

    def test_is_expired_false(self, sample_entry):
        """Test is_expired returns False for non-expired entry."""
        assert sample_entry.is_expired() is False

    def test_is_expired_true(self):
        """Test is_expired returns True for expired entry."""
        now = datetime.now(timezone.utc)
        entry = RevocationEntry(
            token_hash="expired_hash",
            revoked_at=now - timedelta(hours=25),
            expires_at=now - timedelta(hours=1),  # Expired 1 hour ago
        )

        assert entry.is_expired() is True

    def test_to_dict(self, sample_entry):
        """Test to_dict returns correct dictionary."""
        data = sample_entry.to_dict()

        assert data["token_hash"] == "abc123def456"
        assert data["reason"] == "logout"
        assert data["revoked_by"] == "user-123"
        assert data["metadata"] == {"session_id": "sess-456"}
        assert "revoked_at" in data
        assert "expires_at" in data

    def test_to_dict_iso_format(self, sample_entry):
        """Test to_dict returns ISO format dates."""
        data = sample_entry.to_dict()

        # Should be parseable ISO format
        revoked_at = datetime.fromisoformat(data["revoked_at"])
        expires_at = datetime.fromisoformat(data["expires_at"])

        assert revoked_at == sample_entry.revoked_at
        assert expires_at == sample_entry.expires_at


# ============================================================================
# InMemoryRevocationStore Tests
# ============================================================================


class TestInMemoryRevocationStore:
    """Tests for InMemoryRevocationStore class."""

    def test_add_entry(self, memory_store, sample_entry):
        """Test adding a revocation entry."""
        memory_store.add(sample_entry)

        assert memory_store.count() == 1
        assert memory_store.contains(sample_entry.token_hash) is True

    def test_contains_existing(self, memory_store, sample_entry):
        """Test contains returns True for existing token."""
        memory_store.add(sample_entry)

        assert memory_store.contains(sample_entry.token_hash) is True

    def test_contains_nonexistent(self, memory_store):
        """Test contains returns False for non-existent token."""
        assert memory_store.contains("nonexistent_hash") is False

    def test_contains_removes_expired(self, memory_store):
        """Test contains removes expired entries on access."""
        now = datetime.now(timezone.utc)
        expired_entry = RevocationEntry(
            token_hash="expired_hash",
            revoked_at=now - timedelta(hours=2),
            expires_at=now - timedelta(hours=1),  # Expired
        )
        memory_store.add(expired_entry)

        # Contains should return False and remove entry
        assert memory_store.contains("expired_hash") is False
        assert memory_store.count() == 0

    def test_remove_existing(self, memory_store, sample_entry):
        """Test removing an existing entry."""
        memory_store.add(sample_entry)

        result = memory_store.remove(sample_entry.token_hash)

        assert result is True
        assert memory_store.contains(sample_entry.token_hash) is False
        assert memory_store.count() == 0

    def test_remove_nonexistent(self, memory_store):
        """Test removing non-existent entry returns False."""
        result = memory_store.remove("nonexistent_hash")

        assert result is False

    def test_cleanup_expired(self, memory_store):
        """Test cleanup_expired removes expired entries."""
        now = datetime.now(timezone.utc)

        # Add expired entry
        expired_entry = RevocationEntry(
            token_hash="expired_hash",
            revoked_at=now - timedelta(hours=25),
            expires_at=now - timedelta(hours=1),
        )
        memory_store.add(expired_entry)

        # Add valid entry
        valid_entry = RevocationEntry(
            token_hash="valid_hash",
            revoked_at=now,
            expires_at=now + timedelta(hours=24),
        )
        memory_store.add(valid_entry)

        removed = memory_store.cleanup_expired()

        assert removed == 1
        assert memory_store.contains("expired_hash") is False
        assert memory_store.contains("valid_hash") is True

    def test_cleanup_expired_empty_store(self, memory_store):
        """Test cleanup_expired on empty store returns 0."""
        removed = memory_store.cleanup_expired()

        assert removed == 0

    def test_count(self, memory_store):
        """Test count returns correct number of entries."""
        now = datetime.now(timezone.utc)

        assert memory_store.count() == 0

        for i in range(5):
            entry = RevocationEntry(
                token_hash=f"hash_{i}",
                revoked_at=now,
                expires_at=now + timedelta(hours=24),
            )
            memory_store.add(entry)

        assert memory_store.count() == 5

    def test_multiple_entries(self, memory_store):
        """Test storing multiple entries."""
        now = datetime.now(timezone.utc)
        entries = []

        for i in range(10):
            entry = RevocationEntry(
                token_hash=f"token_hash_{i}",
                revoked_at=now,
                expires_at=now + timedelta(hours=24),
                reason=f"reason_{i}",
            )
            entries.append(entry)
            memory_store.add(entry)

        for entry in entries:
            assert memory_store.contains(entry.token_hash) is True

    def test_replace_existing_entry(self, memory_store):
        """Test that adding same token hash replaces entry."""
        now = datetime.now(timezone.utc)

        entry1 = RevocationEntry(
            token_hash="same_hash",
            revoked_at=now,
            expires_at=now + timedelta(hours=1),
            reason="first",
        )
        memory_store.add(entry1)

        entry2 = RevocationEntry(
            token_hash="same_hash",
            revoked_at=now,
            expires_at=now + timedelta(hours=24),
            reason="second",
        )
        memory_store.add(entry2)

        # Count should still be 1
        assert memory_store.count() == 1
        assert memory_store.contains("same_hash") is True

    def test_thread_safety(self, memory_store):
        """Test thread-safe operations."""
        now = datetime.now(timezone.utc)
        errors = []

        def add_entries(prefix, count):
            try:
                for i in range(count):
                    entry = RevocationEntry(
                        token_hash=f"{prefix}_{i}",
                        revoked_at=now,
                        expires_at=now + timedelta(hours=24),
                    )
                    memory_store.add(entry)
            except Exception as e:
                errors.append(e)

        def check_entries(prefix, count):
            try:
                for i in range(count):
                    memory_store.contains(f"{prefix}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_entries, args=("thread1", 50)),
            threading.Thread(target=add_entries, args=("thread2", 50)),
            threading.Thread(target=check_entries, args=("thread1", 50)),
            threading.Thread(target=check_entries, args=("thread2", 50)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_maybe_cleanup_triggers_on_interval(self, memory_store):
        """Test that _maybe_cleanup triggers after interval."""
        # Set last cleanup to far in the past
        memory_store._last_cleanup = time.time() - 600  # 10 minutes ago

        now = datetime.now(timezone.utc)
        entry = RevocationEntry(
            token_hash="trigger_cleanup",
            revoked_at=now,
            expires_at=now + timedelta(hours=24),
        )

        with patch.object(memory_store, "cleanup_expired") as mock_cleanup:
            # Mock threading.Thread to prevent actual thread spawn
            with patch("threading.Thread") as mock_thread:
                mock_thread_instance = MagicMock()
                mock_thread.return_value = mock_thread_instance

                memory_store.add(entry)

                # Thread should have been created for cleanup
                mock_thread.assert_called_once()

    def test_cleanup_interval_configurable(self):
        """Test cleanup interval is configurable."""
        store = InMemoryRevocationStore(cleanup_interval=60.0)
        assert store._cleanup_interval == 60.0


# ============================================================================
# RedisRevocationStore Tests
# ============================================================================


class TestRedisRevocationStore:
    """Tests for RedisRevocationStore class (mocked Redis)."""

    @pytest.fixture
    def redis_store(self):
        """Create a Redis store with mocked client."""
        store = RedisRevocationStore(redis_url="redis://localhost:6379")
        return store

    def test_init_defaults(self):
        """Test RedisRevocationStore initializes with defaults."""
        store = RedisRevocationStore()

        assert store._key_prefix == "aragora:revoked:"
        assert store._client is None

    def test_init_custom_prefix(self):
        """Test RedisRevocationStore with custom key prefix."""
        store = RedisRevocationStore(key_prefix="custom:prefix:")

        assert store._key_prefix == "custom:prefix:"

    def test_key_generation(self, redis_store):
        """Test Redis key generation."""
        key = redis_store._key("token_hash_123")

        assert key == "aragora:revoked:token_hash_123"

    def test_add_entry(self, redis_store, sample_entry):
        """Test adding entry to Redis."""
        mock_client = MagicMock()
        redis_store._client = mock_client

        with patch.object(redis_store, "_get_client", return_value=mock_client):
            redis_store.add(sample_entry)

            mock_client.setex.assert_called_once()
            call_args = mock_client.setex.call_args
            assert sample_entry.token_hash in call_args[0][0]

    def test_add_entry_expired_ttl(self, redis_store):
        """Test adding entry with already expired TTL is skipped."""
        now = datetime.now(timezone.utc)
        expired_entry = RevocationEntry(
            token_hash="expired",
            revoked_at=now - timedelta(hours=2),
            expires_at=now - timedelta(hours=1),  # Already expired
        )

        mock_client = MagicMock()
        redis_store._client = mock_client

        with patch.object(redis_store, "_get_client", return_value=mock_client):
            redis_store.add(expired_entry)

            # setex should not be called for expired entries
            mock_client.setex.assert_not_called()

    def test_contains_existing(self, redis_store):
        """Test contains returns True for existing token."""
        mock_client = MagicMock()
        mock_client.exists.return_value = 1
        redis_store._client = mock_client

        with patch.object(redis_store, "_get_client", return_value=mock_client):
            result = redis_store.contains("existing_hash")

            assert result is True
            mock_client.exists.assert_called_once()

    def test_contains_nonexistent(self, redis_store):
        """Test contains returns False for non-existent token."""
        mock_client = MagicMock()
        mock_client.exists.return_value = 0
        redis_store._client = mock_client

        with patch.object(redis_store, "_get_client", return_value=mock_client):
            result = redis_store.contains("nonexistent_hash")

            assert result is False

    def test_contains_redis_error(self, redis_store):
        """Test contains returns False on Redis error."""
        mock_client = MagicMock()
        mock_client.exists.side_effect = Exception("Redis connection error")
        redis_store._client = mock_client

        with patch.object(redis_store, "_get_client", return_value=mock_client):
            result = redis_store.contains("any_hash")

            assert result is False

    def test_remove_existing(self, redis_store):
        """Test removing existing entry from Redis."""
        mock_client = MagicMock()
        mock_client.delete.return_value = 1
        redis_store._client = mock_client

        with patch.object(redis_store, "_get_client", return_value=mock_client):
            result = redis_store.remove("existing_hash")

            assert result is True
            mock_client.delete.assert_called_once()

    def test_remove_nonexistent(self, redis_store):
        """Test removing non-existent entry returns False."""
        mock_client = MagicMock()
        mock_client.delete.return_value = 0
        redis_store._client = mock_client

        with patch.object(redis_store, "_get_client", return_value=mock_client):
            result = redis_store.remove("nonexistent_hash")

            assert result is False

    def test_remove_redis_error(self, redis_store):
        """Test remove returns False on Redis error."""
        mock_client = MagicMock()
        mock_client.delete.side_effect = Exception("Redis error")
        redis_store._client = mock_client

        with patch.object(redis_store, "_get_client", return_value=mock_client):
            result = redis_store.remove("any_hash")

            assert result is False

    def test_cleanup_expired_returns_zero(self, redis_store):
        """Test cleanup_expired returns 0 (Redis handles TTL)."""
        result = redis_store.cleanup_expired()

        assert result == 0

    def test_count_with_scan(self, redis_store):
        """Test count uses SCAN to count keys."""
        mock_client = MagicMock()
        # Simulate SCAN returning keys in batches
        mock_client.scan.side_effect = [
            (100, [b"key1", b"key2", b"key3"]),
            (0, [b"key4", b"key5"]),
        ]
        redis_store._client = mock_client

        with patch.object(redis_store, "_get_client", return_value=mock_client):
            count = redis_store.count()

            assert count == 5
            assert mock_client.scan.call_count == 2

    def test_count_redis_error(self, redis_store):
        """Test count returns 0 on Redis error."""
        mock_client = MagicMock()
        mock_client.scan.side_effect = Exception("Redis error")
        redis_store._client = mock_client

        with patch.object(redis_store, "_get_client", return_value=mock_client):
            count = redis_store.count()

            assert count == 0

    def test_get_client_creates_redis_connection(self, redis_store):
        """Test _get_client creates Redis connection."""
        pytest.importorskip("redis")

        # Reset any cached client first
        redis_store._client = None

        with patch.dict(os.environ, {}, clear=False):
            with patch("redis.from_url") as mock_from_url:
                mock_client = MagicMock()
                mock_from_url.return_value = mock_client

                client = redis_store._get_client()

                assert client is mock_client
                mock_from_url.assert_called_once_with("redis://localhost:6379")

    def test_get_client_import_error(self, redis_store):
        """Test _get_client raises on import error."""
        redis_store._client = None

        with patch.dict("sys.modules", {"redis": None}):
            with patch("builtins.__import__", side_effect=ImportError("No redis")):
                with pytest.raises(ImportError):
                    redis_store._get_client()


# ============================================================================
# hash_token Tests
# ============================================================================


class TestHashToken:
    """Tests for hash_token function."""

    def test_hash_token_returns_sha256(self):
        """Test hash_token returns SHA-256 hash."""
        token = "my-secret-token"
        expected = hashlib.sha256(token.encode()).hexdigest()

        result = hash_token(token)

        assert result == expected
        assert len(result) == 64  # SHA-256 hex length

    def test_hash_token_consistent(self):
        """Test hash_token returns same hash for same token."""
        token = "consistent-token"

        hash1 = hash_token(token)
        hash2 = hash_token(token)

        assert hash1 == hash2

    def test_hash_token_different_for_different_tokens(self):
        """Test hash_token returns different hashes for different tokens."""
        hash1 = hash_token("token-1")
        hash2 = hash_token("token-2")

        assert hash1 != hash2

    def test_hash_token_empty_string(self):
        """Test hash_token handles empty string."""
        result = hash_token("")

        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_hash_token_unicode(self):
        """Test hash_token handles Unicode tokens."""
        unicode_token = "token_\u4e2d\u6587_\U0001f600"
        result = hash_token(unicode_token)

        expected = hashlib.sha256(unicode_token.encode()).hexdigest()
        assert result == expected


# ============================================================================
# revoke_token Tests
# ============================================================================


class TestRevokeToken:
    """Tests for revoke_token function."""

    def test_revoke_token_basic(self):
        """Test basic token revocation."""
        token = "test-token-to-revoke"

        with patch(
            "aragora.server.middleware.token_revocation.get_revocation_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            entry = revoke_token(token, reason="test")

            assert entry.token_hash == hash_token(token)
            assert entry.reason == "test"
            mock_store.add.assert_called_once()

    def test_revoke_token_with_metadata(self):
        """Test token revocation with metadata."""
        token = "token-with-metadata"
        metadata = {"ip": "192.168.1.1", "user_agent": "Mozilla"}

        with patch(
            "aragora.server.middleware.token_revocation.get_revocation_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            entry = revoke_token(token, metadata=metadata)

            assert entry.metadata == metadata

    def test_revoke_token_custom_ttl(self):
        """Test token revocation with custom TTL."""
        token = "token-custom-ttl"
        ttl_seconds = 7200  # 2 hours

        with patch(
            "aragora.server.middleware.token_revocation.get_revocation_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            entry = revoke_token(token, ttl_seconds=ttl_seconds)

            # Check TTL is approximately correct (within a second)
            expected_ttl = timedelta(seconds=ttl_seconds)
            actual_ttl = entry.expires_at - entry.revoked_at

            assert abs((actual_ttl - expected_ttl).total_seconds()) < 2

    def test_revoke_token_revoked_by(self):
        """Test token revocation with revoked_by field."""
        token = "token-revoked-by"

        with patch(
            "aragora.server.middleware.token_revocation.get_revocation_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            entry = revoke_token(token, revoked_by="admin-user")

            assert entry.revoked_by == "admin-user"

    def test_revoke_token_default_revoked_by(self):
        """Test default revoked_by is 'system'."""
        token = "token-default-revoked-by"

        with patch(
            "aragora.server.middleware.token_revocation.get_revocation_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            entry = revoke_token(token)

            assert entry.revoked_by == "system"

    def test_revoke_token_audit_logging(self):
        """Test token revocation logs audit event."""
        token = "token-audit-log"

        with patch(
            "aragora.server.middleware.token_revocation.get_revocation_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            with patch("aragora.server.middleware.audit_logger.audit_token_revoked") as mock_audit:
                entry = revoke_token(token, reason="security", revoked_by="admin")

                mock_audit.assert_called_once_with(
                    token_hash=entry.token_hash,
                    revoked_by="admin",
                    reason="security",
                )

    def test_revoke_token_audit_import_error(self):
        """Test token revocation handles missing audit logger."""
        token = "token-no-audit"

        with patch(
            "aragora.server.middleware.token_revocation.get_revocation_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            # Should not raise even if audit logger import fails
            entry = revoke_token(token)

            assert entry is not None


# ============================================================================
# is_token_revoked Tests
# ============================================================================


class TestIsTokenRevoked:
    """Tests for is_token_revoked function."""

    def test_is_token_revoked_true(self):
        """Test is_token_revoked returns True for revoked token."""
        token = "revoked-token"

        with patch(
            "aragora.server.middleware.token_revocation.get_revocation_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_store.contains.return_value = True
            mock_get_store.return_value = mock_store

            result = is_token_revoked(token)

            assert result is True
            mock_store.contains.assert_called_once_with(hash_token(token))

    def test_is_token_revoked_false(self):
        """Test is_token_revoked returns False for non-revoked token."""
        token = "valid-token"

        with patch(
            "aragora.server.middleware.token_revocation.get_revocation_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_store.contains.return_value = False
            mock_get_store.return_value = mock_store

            result = is_token_revoked(token)

            assert result is False


# ============================================================================
# unrevoke_token Tests
# ============================================================================


class TestUnrevokeToken:
    """Tests for unrevoke_token function."""

    def test_unrevoke_token_success(self):
        """Test unrevoke_token returns True when token was revoked."""
        token = "token-to-unrevoke"

        with patch(
            "aragora.server.middleware.token_revocation.get_revocation_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_store.remove.return_value = True
            mock_get_store.return_value = mock_store

            result = unrevoke_token(token)

            assert result is True
            mock_store.remove.assert_called_once_with(hash_token(token))

    def test_unrevoke_token_not_revoked(self):
        """Test unrevoke_token returns False when token was not revoked."""
        token = "not-revoked-token"

        with patch(
            "aragora.server.middleware.token_revocation.get_revocation_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_store.remove.return_value = False
            mock_get_store.return_value = mock_store

            result = unrevoke_token(token)

            assert result is False


# ============================================================================
# get_revocation_stats Tests
# ============================================================================


class TestGetRevocationStats:
    """Tests for get_revocation_stats function."""

    def test_get_revocation_stats_memory_store(self):
        """Test stats with in-memory store."""
        with patch(
            "aragora.server.middleware.token_revocation.get_revocation_store"
        ) as mock_get_store:
            mock_store = InMemoryRevocationStore()
            mock_get_store.return_value = mock_store

            stats = get_revocation_stats()

            assert stats["store_type"] == "memory"
            assert stats["revoked_count"] == 0

    def test_get_revocation_stats_redis_store(self):
        """Test stats with Redis store."""
        with patch(
            "aragora.server.middleware.token_revocation.get_revocation_store"
        ) as mock_get_store:
            mock_store = MagicMock(spec=RedisRevocationStore)
            mock_store.count.return_value = 42
            mock_get_store.return_value = mock_store

            # Need to patch isinstance check
            with patch(
                "aragora.server.middleware.token_revocation.isinstance",
                side_effect=lambda obj, cls: cls == RedisRevocationStore,
            ):
                stats = get_revocation_stats()

                assert stats["store_type"] == "redis"
                assert stats["revoked_count"] == 42


# ============================================================================
# get_revocation_store Tests
# ============================================================================


class TestGetRevocationStore:
    """Tests for get_revocation_store function."""

    def test_get_revocation_store_memory_default(self):
        """Test default store is in-memory when no REDIS_URL."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear REDIS_URL if present
            os.environ.pop("REDIS_URL", None)

            store = get_revocation_store()

            assert isinstance(store, InMemoryRevocationStore)

    def test_get_revocation_store_redis_when_available(self):
        """Test Redis store is used when REDIS_URL is set."""
        # Skip if redis package not installed
        try:
            import redis
        except ImportError:
            pytest.skip("redis package not installed")

        with patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379"}):
            with patch("redis.from_url") as mock_redis:
                mock_redis.return_value = MagicMock()

                store = get_revocation_store()

                assert isinstance(store, RedisRevocationStore)

    def test_get_revocation_store_fallback_on_import_error(self):
        """Test fallback to in-memory when redis import fails."""
        with patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379"}):
            # Make Redis import fail during store creation
            import aragora.server.middleware.token_revocation as module

            module._revocation_store = None

            with patch.object(
                RedisRevocationStore, "__init__", side_effect=ImportError("No redis")
            ):
                store = get_revocation_store()

                assert isinstance(store, InMemoryRevocationStore)

    def test_get_revocation_store_cached(self):
        """Test store is cached after first call."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("REDIS_URL", None)

            store1 = get_revocation_store()
            store2 = get_revocation_store()

            assert store1 is store2


# ============================================================================
# Integration Tests
# ============================================================================


class TestTokenRevocationIntegration:
    """Integration tests for token revocation flow."""

    def test_full_revocation_lifecycle(self):
        """Test complete token revocation lifecycle."""
        token = "integration-test-token"

        # Initially not revoked
        assert is_token_revoked(token) is False

        # Revoke the token
        entry = revoke_token(token, reason="logout", revoked_by="user-123")

        # Now should be revoked
        assert is_token_revoked(token) is True
        assert entry.reason == "logout"

        # Unrevoke
        result = unrevoke_token(token)
        assert result is True

        # Should no longer be revoked
        assert is_token_revoked(token) is False

    def test_revoke_multiple_tokens(self):
        """Test revoking multiple tokens."""
        tokens = [f"token-{i}" for i in range(5)]

        for token in tokens:
            revoke_token(token, reason="batch_logout")

        for token in tokens:
            assert is_token_revoked(token) is True

    def test_stats_update_after_revocation(self):
        """Test stats reflect revocation changes."""
        initial_stats = get_revocation_stats()
        initial_count = initial_stats["revoked_count"]

        revoke_token("stats-test-token", reason="test")

        updated_stats = get_revocation_stats()
        assert updated_stats["revoked_count"] == initial_count + 1


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_token(self):
        """Test handling empty token string."""
        with patch(
            "aragora.server.middleware.token_revocation.get_revocation_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            entry = revoke_token("")

            assert entry.token_hash == hash_token("")

    def test_very_long_token(self):
        """Test handling very long token string."""
        long_token = "x" * 10000

        with patch(
            "aragora.server.middleware.token_revocation.get_revocation_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            entry = revoke_token(long_token)

            # Hash should still be 64 chars
            assert len(entry.token_hash) == 64

    def test_unicode_token(self):
        """Test handling Unicode token."""
        unicode_token = "token_\u4e2d\u6587_\U0001f600_emoji"

        with patch(
            "aragora.server.middleware.token_revocation.get_revocation_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            entry = revoke_token(unicode_token)

            assert entry.token_hash == hash_token(unicode_token)

    def test_special_characters_in_reason(self):
        """Test handling special characters in reason."""
        reason = "logout\n\t<script>alert('xss')</script>"

        with patch(
            "aragora.server.middleware.token_revocation.get_revocation_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            entry = revoke_token("token", reason=reason)

            assert entry.reason == reason

    def test_zero_ttl(self):
        """Test token revocation with zero TTL."""
        with patch(
            "aragora.server.middleware.token_revocation.get_revocation_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            entry = revoke_token("zero-ttl-token", ttl_seconds=0)

            # Entry should be created (even if immediately expired)
            assert entry is not None

    def test_negative_ttl(self):
        """Test token revocation with negative TTL."""
        with patch(
            "aragora.server.middleware.token_revocation.get_revocation_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            entry = revoke_token("negative-ttl-token", ttl_seconds=-100)

            # Entry should be created (even if already expired)
            assert entry is not None

    def test_very_large_ttl(self):
        """Test token revocation with very large TTL."""
        huge_ttl = 365 * 24 * 3600 * 100  # 100 years

        with patch(
            "aragora.server.middleware.token_revocation.get_revocation_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            entry = revoke_token("long-ttl-token", ttl_seconds=huge_ttl)

            assert entry is not None

    def test_concurrent_revocation_same_token(self):
        """Test concurrent revocation of same token."""
        token = "concurrent-token"
        errors = []
        results = []

        def revoke_concurrent():
            try:
                entry = revoke_token(token, reason="concurrent")
                results.append(entry)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=revoke_concurrent) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        # All should have same hash
        assert all(r.token_hash == hash_token(token) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
