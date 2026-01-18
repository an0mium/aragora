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
