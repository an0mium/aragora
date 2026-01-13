"""
Tests for the Token Blacklist Storage Backends.

Covers:
- InMemoryBlacklist operations
- SQLiteBlacklist operations
- Token addition and lookup
- Expiration and cleanup
- Size limits and eviction
- Backend selection logic
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from aragora.storage.token_blacklist_store import (
    BlacklistBackend,
    InMemoryBlacklist,
    SQLiteBlacklist,
    get_blacklist_backend,
    set_blacklist_backend,
    MAX_BLACKLIST_SIZE,
)


# =============================================================================
# InMemoryBlacklist Tests
# =============================================================================


class TestInMemoryBlacklist:
    """Tests for the in-memory token blacklist."""

    @pytest.fixture
    def blacklist(self):
        """Create a fresh in-memory blacklist."""
        bl = InMemoryBlacklist(cleanup_interval=300)
        yield bl
        bl.clear()

    def test_add_and_contains(self, blacklist):
        """Test adding a token and checking if it's blacklisted."""
        token_jti = "test_token_123"
        expires_at = time.time() + 3600  # 1 hour from now

        blacklist.add(token_jti, expires_at)

        assert blacklist.contains(token_jti) is True

    def test_contains_nonexistent(self, blacklist):
        """Test checking for non-existent token."""
        assert blacklist.contains("nonexistent_token") is False

    def test_multiple_tokens(self, blacklist):
        """Test adding multiple tokens."""
        expires_at = time.time() + 3600
        tokens = [f"token_{i}" for i in range(10)]

        for token in tokens:
            blacklist.add(token, expires_at)

        for token in tokens:
            assert blacklist.contains(token) is True

    def test_size(self, blacklist):
        """Test getting blacklist size."""
        expires_at = time.time() + 3600

        assert blacklist.size() == 0

        for i in range(5):
            blacklist.add(f"token_{i}", expires_at)

        assert blacklist.size() == 5

    def test_cleanup_expired(self, blacklist):
        """Test cleaning up expired tokens."""
        now = time.time()

        # Add expired token
        blacklist.add("expired_token", now - 100)
        # Add valid token
        blacklist.add("valid_token", now + 3600)

        removed = blacklist.cleanup_expired()

        assert removed == 1
        assert blacklist.contains("expired_token") is False
        assert blacklist.contains("valid_token") is True

    def test_clear(self, blacklist):
        """Test clearing all entries."""
        expires_at = time.time() + 3600

        for i in range(5):
            blacklist.add(f"token_{i}", expires_at)

        assert blacklist.size() == 5

        blacklist.clear()

        assert blacklist.size() == 0

    def test_size_limit_eviction_expired(self, blacklist):
        """Test that size limit evicts expired entries first."""
        now = time.time()

        # Fill up to max size with half expired, half valid
        with patch("aragora.storage.token_blacklist_store.MAX_BLACKLIST_SIZE", 10):
            # Add expired tokens
            for i in range(5):
                blacklist.add(f"expired_{i}", now - 100)
            # Add valid tokens
            for i in range(5):
                blacklist.add(f"valid_{i}", now + 3600)

            # Now add one more - should trigger eviction
            blacklist.add("new_token", now + 3600)

            # Should have evicted some expired tokens
            assert blacklist.contains("new_token") is True
            # Some expired tokens should be gone
            total_expired = sum(1 for i in range(5) if blacklist.contains(f"expired_{i}"))
            assert total_expired < 5

    def test_thread_safety(self, blacklist):
        """Test thread-safe operations."""
        expires_at = time.time() + 3600
        errors = []

        def add_tokens(prefix, count):
            try:
                for i in range(count):
                    blacklist.add(f"{prefix}_{i}", expires_at)
            except Exception as e:
                errors.append(e)

        def check_tokens(prefix, count):
            try:
                for i in range(count):
                    blacklist.contains(f"{prefix}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_tokens, args=("thread1", 100)),
            threading.Thread(target=add_tokens, args=("thread2", 100)),
            threading.Thread(target=check_tokens, args=("thread1", 100)),
            threading.Thread(target=check_tokens, args=("thread2", 100)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# SQLiteBlacklist Tests
# =============================================================================


class TestSQLiteBlacklist:
    """Tests for the SQLite token blacklist."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_blacklist.db"

    @pytest.fixture
    def blacklist(self, temp_db):
        """Create a SQLite blacklist instance."""
        bl = SQLiteBlacklist(temp_db, cleanup_interval=300)
        yield bl
        bl.close()

    def test_add_and_contains(self, blacklist):
        """Test adding a token and checking if it's blacklisted."""
        token_jti = "test_token_sqlite"
        expires_at = time.time() + 3600

        blacklist.add(token_jti, expires_at)

        assert blacklist.contains(token_jti) is True

    def test_contains_nonexistent(self, blacklist):
        """Test checking for non-existent token."""
        assert blacklist.contains("nonexistent_token") is False

    def test_expired_token_not_found(self, blacklist):
        """Test that expired tokens are not found."""
        token_jti = "expired_token_sqlite"
        expires_at = time.time() - 100  # Already expired

        blacklist.add(token_jti, expires_at)

        # Should not find expired token
        assert blacklist.contains(token_jti) is False

    def test_multiple_tokens(self, blacklist):
        """Test adding multiple tokens."""
        expires_at = time.time() + 3600
        tokens = [f"sqlite_token_{i}" for i in range(10)]

        for token in tokens:
            blacklist.add(token, expires_at)

        for token in tokens:
            assert blacklist.contains(token) is True

    def test_size(self, blacklist):
        """Test getting blacklist size."""
        expires_at = time.time() + 3600

        assert blacklist.size() == 0

        for i in range(5):
            blacklist.add(f"sqlite_token_{i}", expires_at)

        assert blacklist.size() == 5

    def test_cleanup_expired(self, blacklist):
        """Test cleaning up expired tokens."""
        now = time.time()

        # Add expired token
        blacklist.add("expired_sqlite", now - 100)
        # Add valid token
        blacklist.add("valid_sqlite", now + 3600)

        removed = blacklist.cleanup_expired()

        assert removed >= 1
        assert blacklist.contains("valid_sqlite") is True

    def test_replace_existing(self, blacklist):
        """Test that adding same token replaces expiration."""
        token_jti = "replace_test"

        # Add with short expiration
        blacklist.add(token_jti, time.time() + 60)
        # Replace with longer expiration
        blacklist.add(token_jti, time.time() + 3600)

        assert blacklist.contains(token_jti) is True
        # Size should still be 1
        assert blacklist.size() == 1

    def test_persistence(self, temp_db):
        """Test that data persists across instances."""
        token_jti = "persistent_token"
        expires_at = time.time() + 3600

        # Create first instance and add token
        bl1 = SQLiteBlacklist(temp_db)
        bl1.add(token_jti, expires_at)
        bl1.close()

        # Create second instance and verify token exists
        bl2 = SQLiteBlacklist(temp_db)
        assert bl2.contains(token_jti) is True
        bl2.close()

    def test_close(self, blacklist):
        """Test closing the connection."""
        # Should not raise
        blacklist.close()
        # Closing again should be safe
        blacklist.close()


# =============================================================================
# Backend Selection Tests
# =============================================================================


class TestBackendSelection:
    """Tests for backend selection logic."""

    @pytest.fixture(autouse=True)
    def reset_backend(self):
        """Reset global backend before each test."""
        import aragora.storage.token_blacklist_store as module

        module._blacklist_backend = None
        yield
        module._blacklist_backend = None

    def test_default_backend_is_sqlite(self):
        """Test that default backend is SQLite."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(os.environ, {"ARAGORA_DATA_DIR": tempfile.mkdtemp()}):
                backend = get_blacklist_backend()
                assert isinstance(backend, SQLiteBlacklist)

    def test_memory_backend_selection(self):
        """Test selecting memory backend."""
        with patch.dict(os.environ, {"ARAGORA_BLACKLIST_BACKEND": "memory"}):
            backend = get_blacklist_backend()
            assert isinstance(backend, InMemoryBlacklist)

    def test_sqlite_backend_selection(self):
        """Test explicitly selecting SQLite backend."""
        with patch.dict(
            os.environ,
            {
                "ARAGORA_BLACKLIST_BACKEND": "sqlite",
                "ARAGORA_DATA_DIR": tempfile.mkdtemp(),
            },
        ):
            backend = get_blacklist_backend()
            assert isinstance(backend, SQLiteBlacklist)

    def test_set_custom_backend(self):
        """Test setting a custom backend."""
        custom_backend = InMemoryBlacklist()

        set_blacklist_backend(custom_backend)

        assert get_blacklist_backend() is custom_backend

    def test_backend_cached(self):
        """Test that backend is cached on first call."""
        with patch.dict(os.environ, {"ARAGORA_BLACKLIST_BACKEND": "memory"}):
            backend1 = get_blacklist_backend()
            backend2 = get_blacklist_backend()

            assert backend1 is backend2


# =============================================================================
# Abstract Base Class Tests
# =============================================================================


class TestBlacklistBackendInterface:
    """Tests for the abstract backend interface."""

    def test_default_size_returns_negative(self):
        """Test that default size() implementation returns -1."""

        class MinimalBackend(BlacklistBackend):
            def add(self, token_jti: str, expires_at: float) -> None:
                pass

            def contains(self, token_jti: str) -> bool:
                return False

            def cleanup_expired(self) -> int:
                return 0

        backend = MinimalBackend()
        assert backend.size() == -1


# =============================================================================
# Edge Cases and Stress Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def memory_blacklist(self):
        """Create a fresh in-memory blacklist."""
        bl = InMemoryBlacklist()
        yield bl
        bl.clear()

    def test_empty_token_id(self, memory_blacklist):
        """Test handling empty token ID."""
        memory_blacklist.add("", time.time() + 3600)
        assert memory_blacklist.contains("") is True

    def test_very_long_token_id(self, memory_blacklist):
        """Test handling very long token ID."""
        long_token = "x" * 10000
        memory_blacklist.add(long_token, time.time() + 3600)
        assert memory_blacklist.contains(long_token) is True

    def test_unicode_token_id(self, memory_blacklist):
        """Test handling Unicode token ID."""
        unicode_token = "token_\u4e2d\u6587_\U0001f600"
        memory_blacklist.add(unicode_token, time.time() + 3600)
        assert memory_blacklist.contains(unicode_token) is True

    def test_zero_expiration(self, memory_blacklist):
        """Test handling zero expiration time."""
        memory_blacklist.add("zero_exp", 0)
        # Zero expiration means already expired
        removed = memory_blacklist.cleanup_expired()
        assert removed >= 1

    def test_negative_expiration(self, memory_blacklist):
        """Test handling negative expiration time."""
        memory_blacklist.add("negative_exp", -1000)
        # Negative expiration means already expired
        removed = memory_blacklist.cleanup_expired()
        assert removed >= 1

    def test_far_future_expiration(self, memory_blacklist):
        """Test handling far future expiration time."""
        far_future = time.time() + (365 * 24 * 3600 * 100)  # 100 years
        memory_blacklist.add("far_future", far_future)
        assert memory_blacklist.contains("far_future") is True
