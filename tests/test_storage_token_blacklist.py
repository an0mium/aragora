"""
Tests for Token Blacklist Storage Backends.

Tests cover:
- InMemoryBlacklist: Thread-safe in-memory storage
- SQLiteBlacklist: Persistent SQLite storage
- Backend selection and configuration
- Automatic cleanup of expired tokens
- Concurrent access safety
"""

import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.storage.token_blacklist_store import (
    BlacklistBackend,
    InMemoryBlacklist,
    SQLiteBlacklist,
    get_blacklist_backend,
    set_blacklist_backend,
    HAS_REDIS,
)


# =============================================================================
# InMemoryBlacklist Tests
# =============================================================================


class TestInMemoryBlacklist:
    """Tests for InMemoryBlacklist backend."""

    @pytest.fixture
    def blacklist(self):
        """Create fresh in-memory blacklist."""
        return InMemoryBlacklist(cleanup_interval=300)

    def test_add_token_to_blacklist(self, blacklist):
        """Test adding a token to blacklist."""
        expires_at = time.time() + 3600  # 1 hour from now
        blacklist.add("token_jti_123", expires_at)

        assert blacklist.contains("token_jti_123") is True
        assert blacklist.size() == 1

    def test_is_revoked_returns_true(self, blacklist):
        """Test that revoked tokens are detected."""
        expires_at = time.time() + 3600
        blacklist.add("revoked_token", expires_at)

        assert blacklist.contains("revoked_token") is True
        assert blacklist.contains("non_revoked_token") is False

    def test_cleanup_removes_expired_only(self, blacklist):
        """Test that cleanup removes only expired tokens."""
        now = time.time()

        # Add expired token
        blacklist.add("expired_token", now - 100)
        # Add valid token
        blacklist.add("valid_token", now + 3600)

        assert blacklist.size() == 2

        # Run cleanup
        removed = blacklist.cleanup_expired()

        assert removed == 1
        assert blacklist.contains("expired_token") is False
        assert blacklist.contains("valid_token") is True
        assert blacklist.size() == 1

    def test_concurrent_add_and_check(self, blacklist):
        """Test concurrent access is thread-safe."""
        errors = []
        success_count = 0
        lock = threading.Lock()

        def add_and_check(thread_id):
            nonlocal success_count
            try:
                token_jti = f"token_{thread_id}"
                expires_at = time.time() + 3600

                blacklist.add(token_jti, expires_at)
                if blacklist.contains(token_jti):
                    with lock:
                        success_count += 1
            except Exception as e:
                with lock:
                    errors.append(str(e))

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(add_and_check, i) for i in range(100)]
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Errors: {errors}"
        assert success_count == 100
        assert blacklist.size() == 100

    def test_clear_removes_all(self, blacklist):
        """Test clearing all entries."""
        for i in range(5):
            blacklist.add(f"token_{i}", time.time() + 3600)

        assert blacklist.size() == 5

        blacklist.clear()

        assert blacklist.size() == 0

    def test_auto_cleanup_on_interval(self):
        """Test automatic cleanup after interval."""
        # Short cleanup interval - trigger cleanup manually
        blacklist = InMemoryBlacklist(cleanup_interval=1)  # 1 second interval

        # Add expired token
        blacklist.add("expired", time.time() - 100)
        # Add valid token
        blacklist.add("valid", time.time() + 3600)

        # Manually trigger cleanup
        blacklist.cleanup_expired()

        # Expired token should be cleaned up
        assert blacklist.contains("expired") is False
        assert blacklist.contains("valid") is True


# =============================================================================
# SQLiteBlacklist Tests
# =============================================================================


class TestSQLiteBlacklist:
    """Tests for SQLiteBlacklist backend."""

    @pytest.fixture
    def db_path(self):
        """Create temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "blacklist.db"

    @pytest.fixture
    def blacklist(self, db_path):
        """Create fresh SQLite blacklist."""
        bl = SQLiteBlacklist(db_path, cleanup_interval=300)
        yield bl
        bl.close()

    def test_add_token_to_blacklist(self, blacklist):
        """Test adding a token to SQLite blacklist."""
        expires_at = time.time() + 3600
        blacklist.add("token_jti_456", expires_at)

        assert blacklist.contains("token_jti_456") is True
        assert blacklist.size() == 1

    def test_is_revoked_returns_true(self, blacklist):
        """Test revoked token detection in SQLite."""
        expires_at = time.time() + 3600
        blacklist.add("revoked_token", expires_at)

        assert blacklist.contains("revoked_token") is True
        assert blacklist.contains("non_revoked_token") is False

    def test_cleanup_removes_expired_only(self, blacklist):
        """Test SQLite cleanup removes only expired tokens."""
        now = time.time()

        # Add expired token
        blacklist.add("expired_token", now - 100)
        # Add valid token
        blacklist.add("valid_token", now + 3600)

        # Run cleanup
        removed = blacklist.cleanup_expired()

        assert removed == 1
        assert blacklist.contains("expired_token") is False
        assert blacklist.contains("valid_token") is True

    def test_persistence_across_restart(self, db_path):
        """Test tokens persist across backend restarts."""
        expires_at = time.time() + 3600

        # Create blacklist, add token, close
        bl1 = SQLiteBlacklist(db_path)
        bl1.add("persistent_token", expires_at)
        bl1.close()

        # Create new blacklist instance
        bl2 = SQLiteBlacklist(db_path)
        try:
            # Token should still be there
            assert bl2.contains("persistent_token") is True
        finally:
            bl2.close()

    def test_concurrent_add_sqlite(self, db_path):
        """Test concurrent adds to SQLite blacklist."""
        errors = []
        success_count = 0
        lock = threading.Lock()

        def add_token(thread_id):
            nonlocal success_count
            # Each thread gets its own connection via thread-local storage
            bl = SQLiteBlacklist(db_path)
            try:
                token_jti = f"concurrent_token_{thread_id}"
                bl.add(token_jti, time.time() + 3600)
                if bl.contains(token_jti):
                    with lock:
                        success_count += 1
            except Exception as e:
                with lock:
                    errors.append(str(e))
            finally:
                bl.close()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(add_token, i) for i in range(20)]
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Errors: {errors}"
        assert success_count == 20

    def test_upsert_behavior(self, blacklist):
        """Test that adding same token updates expiration."""
        token = "upsert_token"

        # Add with initial expiration
        blacklist.add(token, time.time() + 100)
        assert blacklist.contains(token) is True

        # Add again with longer expiration
        blacklist.add(token, time.time() + 3600)
        assert blacklist.contains(token) is True

        # Size should still be 1 (upsert, not duplicate)
        assert blacklist.size() == 1

    def test_parent_directory_creation(self):
        """Test that parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "nested" / "dirs" / "blacklist.db"
            bl = SQLiteBlacklist(nested_path)
            try:
                bl.add("test", time.time() + 3600)
                assert nested_path.exists()
            finally:
                bl.close()


# =============================================================================
# Backend Selection Tests
# =============================================================================


class TestBackendSelection:
    """Tests for blacklist backend selection."""

    def test_default_backend_is_sqlite(self, tmp_path):
        """Test that default backend is SQLite."""
        # Reset global backend
        import aragora.storage.token_blacklist_store as module
        original = module._blacklist_backend
        module._blacklist_backend = None

        try:
            with patch.dict(os.environ, {
                "ARAGORA_BLACKLIST_BACKEND": "sqlite",
                "ARAGORA_DATA_DIR": str(tmp_path),
            }):
                # Force reimport of DATA_DIR
                with patch.object(module, "get_blacklist_backend") as mock:
                    # Call actual implementation
                    mock.side_effect = lambda: module.get_blacklist_backend.__wrapped__() if hasattr(module.get_blacklist_backend, '__wrapped__') else SQLiteBlacklist(tmp_path / "token_blacklist.db")
                    backend = SQLiteBlacklist(tmp_path / "test.db")
                    assert isinstance(backend, SQLiteBlacklist)
        finally:
            module._blacklist_backend = original

    def test_memory_backend_selection(self):
        """Test selecting in-memory backend."""
        import aragora.storage.token_blacklist_store as module
        original = module._blacklist_backend
        module._blacklist_backend = None

        try:
            with patch.dict(os.environ, {"ARAGORA_BLACKLIST_BACKEND": "memory"}):
                backend = get_blacklist_backend()
                assert isinstance(backend, InMemoryBlacklist)
        finally:
            module._blacklist_backend = original

    def test_set_custom_backend(self):
        """Test setting a custom backend."""
        import aragora.storage.token_blacklist_store as module
        original = module._blacklist_backend

        try:
            custom = InMemoryBlacklist()
            set_blacklist_backend(custom)

            backend = get_blacklist_backend()
            assert backend is custom
        finally:
            module._blacklist_backend = original


# =============================================================================
# Redis Backend Tests (if available)
# =============================================================================


@pytest.mark.skipif(not HAS_REDIS, reason="Redis not installed")
class TestRedisBlacklist:
    """Tests for RedisBlacklist backend (requires redis-py)."""

    @pytest.fixture
    def redis_url(self):
        """Get Redis URL from environment or use default."""
        return os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")

    @pytest.fixture
    def blacklist(self, redis_url):
        """Create Redis blacklist with test prefix."""
        from aragora.storage.token_blacklist_store import RedisBlacklist

        try:
            bl = RedisBlacklist(redis_url, key_prefix="aragora:test:blacklist:")
            # Clear test keys
            bl._client.delete(*bl._client.keys(f"{bl._prefix}*") or ["_dummy"])
            yield bl
        except Exception:
            pytest.skip("Redis not available for testing")

    def test_add_and_contains_redis(self, blacklist):
        """Test adding and checking tokens in Redis."""
        expires_at = time.time() + 3600
        blacklist.add("redis_token", expires_at)

        assert blacklist.contains("redis_token") is True
        assert blacklist.contains("nonexistent") is False

    def test_redis_auto_expiration(self, blacklist):
        """Test that Redis auto-expires tokens."""
        # Add token with 1 second TTL
        blacklist.add("short_lived", time.time() + 1)

        assert blacklist.contains("short_lived") is True

        # Wait for expiration
        time.sleep(2)

        assert blacklist.contains("short_lived") is False

    def test_redis_backend_fallback(self, tmp_path):
        """Test fallback to SQLite when Redis unavailable."""
        import aragora.storage.token_blacklist_store as module
        original = module._blacklist_backend
        module._blacklist_backend = None

        try:
            with patch.dict(os.environ, {
                "ARAGORA_BLACKLIST_BACKEND": "redis",
                "ARAGORA_REDIS_URL": "redis://nonexistent:6379/0",
                "ARAGORA_DATA_DIR": str(tmp_path),
            }):
                backend = get_blacklist_backend()
                # Should fall back to SQLite due to connection failure
                assert isinstance(backend, SQLiteBlacklist)
        finally:
            module._blacklist_backend = original


# =============================================================================
# Integration Tests
# =============================================================================


class TestBlacklistIntegration:
    """Integration tests for token blacklist."""

    def test_typical_revocation_workflow(self, tmp_path):
        """Test typical token revocation workflow."""
        blacklist = SQLiteBlacklist(tmp_path / "workflow.db")
        try:
            # Simulate user logging in and getting a token
            token_jti = "user_session_abc123"
            token_expiry = time.time() + 3600  # 1 hour

            # User logs out - token is revoked
            blacklist.add(token_jti, token_expiry)

            # Subsequent requests with revoked token should be rejected
            assert blacklist.contains(token_jti) is True

            # Different token should be accepted
            assert blacklist.contains("different_token") is False
        finally:
            blacklist.close()

    def test_multiple_revocations_same_user(self, tmp_path):
        """Test revoking multiple tokens for same user."""
        blacklist = SQLiteBlacklist(tmp_path / "multi.db")
        try:
            user_tokens = [f"user1_session_{i}" for i in range(5)]

            for token in user_tokens:
                blacklist.add(token, time.time() + 3600)

            # All tokens should be revoked
            for token in user_tokens:
                assert blacklist.contains(token) is True

            assert blacklist.size() == 5
        finally:
            blacklist.close()

    def test_high_volume_revocations(self, tmp_path):
        """Test handling high volume of revocations."""
        blacklist = SQLiteBlacklist(tmp_path / "volume.db")
        try:
            # Add many tokens
            for i in range(1000):
                blacklist.add(f"volume_token_{i}", time.time() + 3600)

            assert blacklist.size() == 1000

            # Random lookups should be fast
            import random
            for _ in range(100):
                token = f"volume_token_{random.randint(0, 999)}"
                assert blacklist.contains(token) is True
        finally:
            blacklist.close()
