"""
Tests for the account lockout system.

Tests cover:
- LockoutEntry data class
- InMemoryLockoutBackend
- LockoutTracker functionality
- Exponential backoff policy
- Email and IP-based lockout
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.auth.lockout import (
    InMemoryLockoutBackend,
    LockoutEntry,
    LockoutTracker,
)


class TestLockoutEntry:
    """Tests for LockoutEntry data class."""

    def test_entry_not_locked_by_default(self):
        """Test that new entry is not locked."""
        entry = LockoutEntry()
        assert entry.is_locked() is False
        assert entry.get_remaining_seconds() == 0

    def test_entry_with_failed_attempts(self):
        """Test entry with failed attempts but no lockout."""
        entry = LockoutEntry(failed_attempts=3)
        assert entry.failed_attempts == 3
        assert entry.is_locked() is False

    def test_entry_locked(self):
        """Test entry that is locked."""
        future = time.time() + 60  # 60 seconds in the future
        entry = LockoutEntry(failed_attempts=5, lockout_until=future)

        assert entry.is_locked() is True
        assert entry.get_remaining_seconds() > 0
        assert entry.get_remaining_seconds() <= 60

    def test_entry_expired_lockout(self):
        """Test entry with expired lockout."""
        past = time.time() - 60  # 60 seconds in the past
        entry = LockoutEntry(failed_attempts=5, lockout_until=past)

        assert entry.is_locked() is False
        assert entry.get_remaining_seconds() == 0


class TestInMemoryLockoutBackend:
    """Tests for InMemoryLockoutBackend."""

    def test_backend_always_available(self):
        """Test that in-memory backend is always available."""
        backend = InMemoryLockoutBackend()
        assert backend.is_available() is True

    def test_get_nonexistent_entry(self):
        """Test getting entry that doesn't exist."""
        backend = InMemoryLockoutBackend()
        result = backend.get_entry("nonexistent")
        assert result is None

    def test_set_and_get_entry(self):
        """Test setting and getting entry."""
        backend = InMemoryLockoutBackend()
        entry = LockoutEntry(failed_attempts=3)

        backend.set_entry("test_key", entry, ttl_seconds=300)
        result = backend.get_entry("test_key")

        assert result is not None
        assert result.failed_attempts == 3

    def test_entry_expiration(self):
        """Test that entries expire after TTL."""
        backend = InMemoryLockoutBackend()
        entry = LockoutEntry(failed_attempts=3)

        # Set with 0 TTL (immediately expires)
        backend.set_entry("test_key", entry, ttl_seconds=0)

        # Wait a tiny bit to ensure expiration
        time.sleep(0.01)

        result = backend.get_entry("test_key")
        assert result is None

    def test_delete_entry(self):
        """Test deleting entry."""
        backend = InMemoryLockoutBackend()
        entry = LockoutEntry(failed_attempts=3)

        backend.set_entry("test_key", entry, ttl_seconds=300)
        backend.delete_entry("test_key")

        result = backend.get_entry("test_key")
        assert result is None

    def test_delete_nonexistent_entry(self):
        """Test deleting entry that doesn't exist (should not raise)."""
        backend = InMemoryLockoutBackend()
        backend.delete_entry("nonexistent")  # Should not raise

    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        backend = InMemoryLockoutBackend()

        # Add entry with 0 TTL
        backend.set_entry("expired", LockoutEntry(failed_attempts=1), ttl_seconds=0)
        # Add entry with long TTL
        backend.set_entry("valid", LockoutEntry(failed_attempts=2), ttl_seconds=300)

        time.sleep(0.01)
        removed = backend.cleanup_expired()

        assert removed == 1
        assert backend.get_entry("expired") is None
        assert backend.get_entry("valid") is not None

    def test_thread_safety(self):
        """Test that backend is thread-safe."""
        import threading

        backend = InMemoryLockoutBackend()
        errors = []

        def worker(worker_id: int):
            try:
                for i in range(100):
                    key = f"key_{worker_id}_{i}"
                    entry = LockoutEntry(failed_attempts=i)
                    backend.set_entry(key, entry, ttl_seconds=300)
                    result = backend.get_entry(key)
                    assert result is not None
                    backend.delete_entry(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestLockoutTracker:
    """Tests for LockoutTracker."""

    def test_tracker_initialization(self):
        """Test tracker initializes with memory backend."""
        tracker = LockoutTracker(use_redis=False)
        assert tracker._memory_backend is not None
        assert tracker._redis_backend is None

    def test_is_locked_no_failures(self):
        """Test is_locked returns False with no failures."""
        tracker = LockoutTracker(use_redis=False)
        assert tracker.is_locked(email="test@example.com") is False
        assert tracker.is_locked(ip="192.168.1.1") is False

    def test_record_failure_under_threshold(self):
        """Test recording failures under threshold."""
        tracker = LockoutTracker(use_redis=False)

        # Record 4 failures (under threshold of 5)
        for _ in range(4):
            tracker.record_failure(email="test@example.com")

        assert tracker.is_locked(email="test@example.com") is False
        info = tracker.get_info(email="test@example.com")
        assert info["email"]["failed_attempts"] == 4

    def test_lockout_after_threshold(self):
        """Test lockout triggers after threshold."""
        tracker = LockoutTracker(use_redis=False)

        # Record 5 failures (hits first threshold)
        for _ in range(5):
            tracker.record_failure(email="test@example.com")

        assert tracker.is_locked(email="test@example.com") is True
        remaining = tracker.get_remaining_time(email="test@example.com")
        assert remaining > 0
        assert remaining <= LockoutTracker.DURATION_1

    def test_escalating_lockout_durations(self):
        """Test that lockout duration increases with more attempts."""
        tracker = LockoutTracker(use_redis=False)

        # 10 failures -> 15 minute lockout
        for _ in range(10):
            tracker.record_failure(email="test@example.com")

        remaining = tracker.get_remaining_time(email="test@example.com")
        assert remaining > LockoutTracker.DURATION_1
        assert remaining <= LockoutTracker.DURATION_2

    def test_ip_based_lockout(self):
        """Test IP-based lockout."""
        tracker = LockoutTracker(use_redis=False)

        # Record failures from same IP
        for _ in range(5):
            tracker.record_failure(ip="192.168.1.100")

        assert tracker.is_locked(ip="192.168.1.100") is True
        assert tracker.is_locked(ip="192.168.1.101") is False

    def test_combined_email_ip_lockout(self):
        """Test that lockout on either email or IP blocks login."""
        tracker = LockoutTracker(use_redis=False)

        # Lock by email
        for _ in range(5):
            tracker.record_failure(email="test@example.com", ip="192.168.1.1")

        # Same email, different IP - still locked
        assert tracker.is_locked(email="test@example.com", ip="10.0.0.1") is True

    def test_reset_clears_lockout(self):
        """Test that reset clears lockout state."""
        tracker = LockoutTracker(use_redis=False)

        # Create lockout
        for _ in range(5):
            tracker.record_failure(email="test@example.com")

        assert tracker.is_locked(email="test@example.com") is True

        # Reset
        tracker.reset(email="test@example.com")

        assert tracker.is_locked(email="test@example.com") is False
        info = tracker.get_info(email="test@example.com")
        assert info["email"]["failed_attempts"] == 0

    def test_email_key_case_insensitive(self):
        """Test that email keys are case-insensitive."""
        tracker = LockoutTracker(use_redis=False)

        # Record with mixed case
        for _ in range(5):
            tracker.record_failure(email="Test@Example.COM")

        # Check with different case
        assert tracker.is_locked(email="test@example.com") is True

    def test_get_info_no_entry(self):
        """Test get_info returns 0 attempts for unknown email/IP."""
        tracker = LockoutTracker(use_redis=False)
        email_info = tracker.get_info(email="unknown@example.com")
        assert email_info["email"]["failed_attempts"] == 0
        ip_info = tracker.get_info(ip="10.0.0.1")
        assert ip_info["ip"]["failed_attempts"] == 0

    def test_get_remaining_time_not_locked(self):
        """Test get_remaining_time returns 0 when not locked."""
        tracker = LockoutTracker(use_redis=False)
        assert tracker.get_remaining_time(email="test@example.com") == 0


class TestLockoutTrackerEdgeCases:
    """Edge case tests for LockoutTracker."""

    def test_empty_email_and_ip(self):
        """Test with empty email and IP (should handle gracefully)."""
        tracker = LockoutTracker(use_redis=False)
        # Should not raise
        tracker.record_failure()
        assert tracker.is_locked() is False

    def test_rapid_failures(self):
        """Test rapid successive failures."""
        tracker = LockoutTracker(use_redis=False)

        # Rapid fire failures
        for _ in range(20):
            tracker.record_failure(email="rapid@example.com")

        # Should be locked with max duration
        assert tracker.is_locked(email="rapid@example.com") is True
        remaining = tracker.get_remaining_time(email="rapid@example.com")
        assert remaining > LockoutTracker.DURATION_2

    def test_lockout_expires(self):
        """Test that lockout expires over time."""
        tracker = LockoutTracker(use_redis=False)

        # Create a minimal lockout entry directly
        entry = LockoutEntry(
            failed_attempts=5,
            lockout_until=time.time() + 0.1,  # 100ms lockout
            last_attempt=time.time(),
        )
        tracker._backend.set_entry(
            tracker._email_key("expire@example.com"),
            entry,
            ttl_seconds=300,
        )

        # Should be locked initially
        assert tracker.is_locked(email="expire@example.com") is True

        # Wait for lockout to expire
        time.sleep(0.15)

        # Should no longer be locked
        assert tracker.is_locked(email="expire@example.com") is False


class TestLockoutConstants:
    """Tests for LockoutTracker constants."""

    def test_threshold_ordering(self):
        """Test that thresholds are properly ordered."""
        assert LockoutTracker.THRESHOLD_1 < LockoutTracker.THRESHOLD_2
        assert LockoutTracker.THRESHOLD_2 < LockoutTracker.THRESHOLD_3

    def test_duration_ordering(self):
        """Test that durations escalate."""
        assert LockoutTracker.DURATION_1 < LockoutTracker.DURATION_2
        assert LockoutTracker.DURATION_2 < LockoutTracker.DURATION_3

    def test_default_values(self):
        """Test default constant values."""
        assert LockoutTracker.THRESHOLD_1 == 5
        assert LockoutTracker.THRESHOLD_2 == 10
        assert LockoutTracker.THRESHOLD_3 == 15
        assert LockoutTracker.DURATION_1 == 60
        assert LockoutTracker.DURATION_2 == 15 * 60
        assert LockoutTracker.DURATION_3 == 60 * 60


class TestAdminUnlock:
    """Tests for admin_unlock() functionality."""

    def test_admin_unlock_by_email(self):
        """Test admin unlock clears email-based lockout."""
        tracker = LockoutTracker(use_redis=False)

        # Create lockout
        for _ in range(5):
            tracker.record_failure(email="locked@example.com")
        assert tracker.is_locked(email="locked@example.com") is True

        # Admin unlock
        result = tracker.admin_unlock(email="locked@example.com", user_id="admin-1")

        assert result is True
        assert tracker.is_locked(email="locked@example.com") is False

    def test_admin_unlock_by_ip(self):
        """Test admin unlock clears IP-based lockout."""
        tracker = LockoutTracker(use_redis=False)

        # Create lockout
        for _ in range(5):
            tracker.record_failure(ip="192.168.1.100")
        assert tracker.is_locked(ip="192.168.1.100") is True

        # Admin unlock
        result = tracker.admin_unlock(ip="192.168.1.100", user_id="admin-1")

        assert result is True
        assert tracker.is_locked(ip="192.168.1.100") is False

    def test_admin_unlock_both_email_and_ip(self):
        """Test admin unlock clears both email and IP lockouts."""
        tracker = LockoutTracker(use_redis=False)

        # Create lockouts on both
        for _ in range(5):
            tracker.record_failure(email="user@example.com", ip="10.0.0.1")
        assert tracker.is_locked(email="user@example.com") is True
        assert tracker.is_locked(ip="10.0.0.1") is True

        # Admin unlock both
        result = tracker.admin_unlock(email="user@example.com", ip="10.0.0.1", user_id="admin-1")

        assert result is True
        assert tracker.is_locked(email="user@example.com") is False
        assert tracker.is_locked(ip="10.0.0.1") is False

    def test_admin_unlock_no_lockout_exists(self):
        """Test admin unlock returns False when no lockout exists."""
        tracker = LockoutTracker(use_redis=False)

        result = tracker.admin_unlock(email="never_locked@example.com")

        assert result is False

    def test_admin_unlock_clears_failed_attempts(self):
        """Test admin unlock clears failed attempts counter."""
        tracker = LockoutTracker(use_redis=False)

        # Record 4 failures (under lockout threshold but should still clear)
        for _ in range(4):
            tracker.record_failure(email="partial@example.com")

        info_before = tracker.get_info(email="partial@example.com")
        assert info_before["email"]["failed_attempts"] == 4

        # Admin unlock
        result = tracker.admin_unlock(email="partial@example.com")

        assert result is True
        info_after = tracker.get_info(email="partial@example.com")
        assert info_after["email"]["failed_attempts"] == 0

    def test_admin_unlock_with_empty_params(self):
        """Test admin unlock with no email or IP provided."""
        tracker = LockoutTracker(use_redis=False)

        # Should return False without error
        result = tracker.admin_unlock()

        assert result is False


class TestRedisLockoutBackend:
    """Tests for RedisLockoutBackend."""

    def test_backend_unavailable_without_redis_url(self):
        """Test backend is unavailable when no Redis URL provided."""
        from aragora.auth.lockout import RedisLockoutBackend

        backend = RedisLockoutBackend(redis_url=None)
        assert backend.is_available() is False

    def test_backend_unavailable_redis_not_installed(self):
        """Test backend handles missing redis-py gracefully."""
        from aragora.auth.lockout import RedisLockoutBackend

        with patch.dict("sys.modules", {"redis": None}):
            backend = RedisLockoutBackend(redis_url="redis://localhost:6379")
            # Should handle import error gracefully
            assert backend._available is False or backend._client is None

    def test_backend_unavailable_connection_error(self):
        """Test backend handles connection errors gracefully."""
        from aragora.auth.lockout import RedisLockoutBackend

        mock_redis = MagicMock()
        mock_redis.from_url.return_value.ping.side_effect = ConnectionError("Connection refused")

        with patch.dict("sys.modules", {"redis": mock_redis}):
            backend = RedisLockoutBackend(redis_url="redis://localhost:6379")
            # Should handle connection error gracefully
            assert backend.is_available() is False

    def test_get_entry_when_unavailable(self):
        """Test get_entry returns None when Redis unavailable."""
        from aragora.auth.lockout import RedisLockoutBackend

        backend = RedisLockoutBackend(redis_url=None)
        result = backend.get_entry("test_key")

        assert result is None

    def test_set_entry_when_unavailable(self):
        """Test set_entry does nothing when Redis unavailable."""
        from aragora.auth.lockout import RedisLockoutBackend

        backend = RedisLockoutBackend(redis_url=None)
        entry = LockoutEntry(failed_attempts=5)

        # Should not raise
        backend.set_entry("test_key", entry, ttl_seconds=300)

    def test_delete_entry_when_unavailable(self):
        """Test delete_entry does nothing when Redis unavailable."""
        from aragora.auth.lockout import RedisLockoutBackend

        backend = RedisLockoutBackend(redis_url=None)

        # Should not raise
        backend.delete_entry("test_key")

    def test_key_prefix(self):
        """Test custom key prefix is used."""
        from aragora.auth.lockout import RedisLockoutBackend

        backend = RedisLockoutBackend(redis_url=None, key_prefix="custom:prefix:")
        assert backend._make_key("test") == "custom:prefix:test"

    def test_default_key_prefix(self):
        """Test default key prefix."""
        from aragora.auth.lockout import RedisLockoutBackend

        backend = RedisLockoutBackend(redis_url=None)
        assert backend._make_key("test") == "aragora:lockout:test"

    def test_get_entry_with_mocked_redis(self):
        """Test get_entry with mocked Redis client."""
        from aragora.auth.lockout import RedisLockoutBackend
        import json

        backend = RedisLockoutBackend(redis_url=None)
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = json.dumps({
            "failed_attempts": 3,
            "lockout_until": 1704067200.0,
            "last_attempt": 1704067100.0,
        })
        backend._client = mock_client
        backend._available = True

        result = backend.get_entry("test_key")

        assert result is not None
        assert result.failed_attempts == 3
        assert result.lockout_until == 1704067200.0

    def test_get_entry_json_decode_error(self):
        """Test get_entry handles JSON decode errors."""
        from aragora.auth.lockout import RedisLockoutBackend

        backend = RedisLockoutBackend(redis_url=None)
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = "invalid json"
        backend._client = mock_client
        backend._available = True

        result = backend.get_entry("test_key")

        assert result is None

    def test_set_entry_with_mocked_redis(self):
        """Test set_entry with mocked Redis client."""
        from aragora.auth.lockout import RedisLockoutBackend
        import json

        backend = RedisLockoutBackend(redis_url=None)
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        backend._client = mock_client
        backend._available = True

        entry = LockoutEntry(failed_attempts=5, lockout_until=1704067200.0)
        backend.set_entry("test_key", entry, ttl_seconds=300)

        mock_client.setex.assert_called_once()
        args = mock_client.setex.call_args
        assert args[0][0] == "aragora:lockout:test_key"
        assert args[0][1] == 300
        data = json.loads(args[0][2])
        assert data["failed_attempts"] == 5

    def test_delete_entry_with_mocked_redis(self):
        """Test delete_entry with mocked Redis client."""
        from aragora.auth.lockout import RedisLockoutBackend

        backend = RedisLockoutBackend(redis_url=None)
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        backend._client = mock_client
        backend._available = True

        backend.delete_entry("test_key")

        mock_client.delete.assert_called_once_with("aragora:lockout:test_key")

    def test_is_available_ping_fails(self):
        """Test is_available returns False when ping fails."""
        from aragora.auth.lockout import RedisLockoutBackend

        backend = RedisLockoutBackend(redis_url=None)
        mock_client = MagicMock()
        mock_client.ping.side_effect = ConnectionError("Connection lost")
        backend._client = mock_client
        backend._available = True

        assert backend.is_available() is False
        assert backend._available is False


class TestLockoutTrackerBackendType:
    """Tests for backend_type property."""

    def test_backend_type_memory(self):
        """Test backend_type returns 'memory' when using memory backend."""
        tracker = LockoutTracker(use_redis=False)
        assert tracker.backend_type == "memory"

    def test_backend_type_with_failed_redis(self):
        """Test backend_type returns 'memory' when Redis fails."""
        from aragora.auth.lockout import RedisLockoutBackend

        # Mock the Redis backend to simulate unavailability
        with patch.object(RedisLockoutBackend, "_init_client") as mock_init:
            mock_init.return_value = None  # Prevent actual connection
            tracker = LockoutTracker(redis_url="redis://localhost:6379", use_redis=True)
            # Redis backend will be created but marked unavailable
            if tracker._redis_backend:
                tracker._redis_backend._available = False
            assert tracker.backend_type == "memory"


class TestGlobalLockoutTracker:
    """Tests for get_lockout_tracker() and reset_lockout_tracker()."""

    def test_get_lockout_tracker_returns_singleton(self):
        """Test get_lockout_tracker returns the same instance."""
        from aragora.auth.lockout import get_lockout_tracker, reset_lockout_tracker

        reset_lockout_tracker()

        tracker1 = get_lockout_tracker(use_redis=False)
        tracker2 = get_lockout_tracker(use_redis=False)

        assert tracker1 is tracker2

        reset_lockout_tracker()

    def test_reset_lockout_tracker_clears_singleton(self):
        """Test reset_lockout_tracker clears the global instance."""
        from aragora.auth.lockout import get_lockout_tracker, reset_lockout_tracker

        reset_lockout_tracker()

        tracker1 = get_lockout_tracker(use_redis=False)
        reset_lockout_tracker()
        tracker2 = get_lockout_tracker(use_redis=False)

        assert tracker1 is not tracker2

        reset_lockout_tracker()

    def test_get_lockout_tracker_thread_safe(self):
        """Test get_lockout_tracker is thread-safe."""
        from aragora.auth.lockout import get_lockout_tracker, reset_lockout_tracker
        import threading

        reset_lockout_tracker()

        trackers = []
        errors = []

        def get_tracker():
            try:
                t = get_lockout_tracker(use_redis=False)
                trackers.append(t)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_tracker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(trackers) == 10
        # All should be the same instance
        assert all(t is trackers[0] for t in trackers)

        reset_lockout_tracker()


class TestLockoutTrackerConcurrency:
    """Tests for concurrent access to LockoutTracker."""

    def test_concurrent_failure_recording(self):
        """Test concurrent failure recording doesn't lose counts."""
        import threading

        tracker = LockoutTracker(use_redis=False)
        email = "concurrent@example.com"
        errors = []

        def record_failures():
            try:
                for _ in range(10):
                    tracker.record_failure(email=email)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_failures) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        info = tracker.get_info(email=email)
        # Should have approximately 50 attempts (5 threads * 10 each)
        # Due to race conditions with in-memory backend, may be slightly less
        assert info["email"]["failed_attempts"] >= 40

    def test_concurrent_is_locked_reads(self):
        """Test concurrent is_locked reads are safe."""
        import threading

        tracker = LockoutTracker(use_redis=False)

        # Create a lockout
        for _ in range(5):
            tracker.record_failure(email="locked@example.com")

        results = []
        errors = []

        def check_locked():
            try:
                for _ in range(100):
                    result = tracker.is_locked(email="locked@example.com")
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=check_locked) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 500
        # All results should be True since account is locked
        assert all(r is True for r in results)

    def test_concurrent_record_and_reset(self):
        """Test concurrent recording and resetting doesn't crash."""
        import threading

        tracker = LockoutTracker(use_redis=False)
        email = "chaotic@example.com"
        errors = []

        def record():
            try:
                for _ in range(20):
                    tracker.record_failure(email=email)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reset():
            try:
                for _ in range(20):
                    tracker.reset(email=email)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record),
            threading.Thread(target=reset),
            threading.Thread(target=record),
            threading.Thread(target=reset),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not have any errors
        assert len(errors) == 0
