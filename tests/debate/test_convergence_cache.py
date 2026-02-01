"""
Tests for the similarity cache manager cleanup functionality.

Tests cover:
- Stale cache cleanup based on TTL
- Max cache limit enforcement
- Cache timestamp tracking
- Oldest cache eviction when at capacity
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from aragora.debate.convergence import (
    CACHE_MANAGER_TTL_SECONDS,
    MAX_SIMILARITY_CACHES,
    PairwiseSimilarityCache,
    cleanup_similarity_cache,
    cleanup_stale_similarity_caches,
    get_pairwise_similarity_cache,
)


class TestSimilarityCacheTimestampTracking:
    """Tests for cache timestamp tracking."""

    def setup_method(self):
        """Clear global caches before each test."""
        # Import the global dicts to clear them
        from aragora.debate import convergence

        with convergence._similarity_cache_lock:
            # Clear all existing caches
            for session_id in list(convergence._similarity_cache_manager.keys()):
                convergence._similarity_cache_manager[session_id].clear()
            convergence._similarity_cache_manager.clear()
            convergence._similarity_cache_timestamps.clear()

    def teardown_method(self):
        """Clear global caches after each test."""
        from aragora.debate import convergence

        with convergence._similarity_cache_lock:
            for session_id in list(convergence._similarity_cache_manager.keys()):
                convergence._similarity_cache_manager[session_id].clear()
            convergence._similarity_cache_manager.clear()
            convergence._similarity_cache_timestamps.clear()

    def test_cache_timestamp_tracking(self):
        """Test that cache creation timestamps are tracked."""
        from aragora.debate import convergence

        # Create a cache
        session_id = "test_timestamp_session"
        before_time = time.time()
        cache = get_pairwise_similarity_cache(session_id)
        after_time = time.time()

        # Verify timestamp was recorded
        assert session_id in convergence._similarity_cache_timestamps
        timestamp = convergence._similarity_cache_timestamps[session_id]
        assert before_time <= timestamp <= after_time

        # Verify cache was created
        assert session_id in convergence._similarity_cache_manager
        assert cache.session_id == session_id

    def test_cache_timestamp_updated_on_access(self):
        """Test that timestamp is updated when cache is accessed."""
        from aragora.debate import convergence

        session_id = "test_access_session"

        # Create cache with initial timestamp
        get_pairwise_similarity_cache(session_id)
        initial_timestamp = convergence._similarity_cache_timestamps[session_id]

        # Wait a bit and access again
        time.sleep(0.01)
        get_pairwise_similarity_cache(session_id)
        updated_timestamp = convergence._similarity_cache_timestamps[session_id]

        # Timestamp should be updated
        assert updated_timestamp > initial_timestamp

    def test_cleanup_removes_timestamp(self):
        """Test that cleanup_similarity_cache removes the timestamp."""
        from aragora.debate import convergence

        session_id = "test_cleanup_timestamp"
        get_pairwise_similarity_cache(session_id)

        # Verify it exists
        assert session_id in convergence._similarity_cache_timestamps
        assert session_id in convergence._similarity_cache_manager

        # Cleanup
        cleanup_similarity_cache(session_id)

        # Both should be removed
        assert session_id not in convergence._similarity_cache_timestamps
        assert session_id not in convergence._similarity_cache_manager


class TestCleanupStaleCaches:
    """Tests for cleanup_stale_similarity_caches function."""

    def setup_method(self):
        """Clear global caches before each test."""
        from aragora.debate import convergence

        with convergence._similarity_cache_lock:
            for session_id in list(convergence._similarity_cache_manager.keys()):
                convergence._similarity_cache_manager[session_id].clear()
            convergence._similarity_cache_manager.clear()
            convergence._similarity_cache_timestamps.clear()

    def teardown_method(self):
        """Clear global caches after each test."""
        from aragora.debate import convergence

        with convergence._similarity_cache_lock:
            for session_id in list(convergence._similarity_cache_manager.keys()):
                convergence._similarity_cache_manager[session_id].clear()
            convergence._similarity_cache_manager.clear()
            convergence._similarity_cache_timestamps.clear()

    def test_cleanup_stale_caches(self):
        """Test that stale caches are cleaned up based on TTL."""
        from aragora.debate import convergence

        # Create caches with different ages
        old_session = "old_session"
        new_session = "new_session"

        # Create old cache with fake old timestamp
        get_pairwise_similarity_cache(old_session)
        convergence._similarity_cache_timestamps[old_session] = time.time() - 7200  # 2 hours ago

        # Create new cache
        get_pairwise_similarity_cache(new_session)

        # Verify both exist
        assert old_session in convergence._similarity_cache_manager
        assert new_session in convergence._similarity_cache_manager

        # Cleanup with 1 hour TTL (default)
        cleaned = cleanup_stale_similarity_caches(max_age_seconds=3600)

        # Old cache should be removed, new should remain
        assert cleaned == 1
        assert old_session not in convergence._similarity_cache_manager
        assert old_session not in convergence._similarity_cache_timestamps
        assert new_session in convergence._similarity_cache_manager
        assert new_session in convergence._similarity_cache_timestamps

    def test_cleanup_with_custom_max_age(self):
        """Test cleanup with custom max age parameter."""
        from aragora.debate import convergence

        session_id = "custom_age_session"
        get_pairwise_similarity_cache(session_id)

        # Set timestamp to 10 seconds ago
        convergence._similarity_cache_timestamps[session_id] = time.time() - 10

        # Cleanup with 5 second max age
        cleaned = cleanup_stale_similarity_caches(max_age_seconds=5)
        assert cleaned == 1
        assert session_id not in convergence._similarity_cache_manager

    def test_cleanup_returns_zero_when_no_stale(self):
        """Test that cleanup returns 0 when no stale caches exist."""
        from aragora.debate import convergence

        # Create a fresh cache
        get_pairwise_similarity_cache("fresh_session")

        # All caches are fresh, should clean nothing
        cleaned = cleanup_stale_similarity_caches()
        assert cleaned == 0

    def test_cleanup_with_empty_manager(self):
        """Test cleanup when cache manager is empty."""
        cleaned = cleanup_stale_similarity_caches()
        assert cleaned == 0


class TestMaxCachesEnforced:
    """Tests for max cache limit enforcement."""

    def setup_method(self):
        """Clear global caches before each test."""
        from aragora.debate import convergence

        with convergence._similarity_cache_lock:
            for session_id in list(convergence._similarity_cache_manager.keys()):
                convergence._similarity_cache_manager[session_id].clear()
            convergence._similarity_cache_manager.clear()
            convergence._similarity_cache_timestamps.clear()

    def teardown_method(self):
        """Clear global caches after each test."""
        from aragora.debate import convergence

        with convergence._similarity_cache_lock:
            for session_id in list(convergence._similarity_cache_manager.keys()):
                convergence._similarity_cache_manager[session_id].clear()
            convergence._similarity_cache_manager.clear()
            convergence._similarity_cache_timestamps.clear()

    def test_max_caches_enforced(self):
        """Test that max cache limit is enforced."""
        from aragora.debate import convergence

        # Temporarily reduce max caches for testing
        with patch.object(convergence, "MAX_SIMILARITY_CACHES", 5):
            # Create max number of caches
            for i in range(5):
                get_pairwise_similarity_cache(f"session_{i}")
                # Add small delay to ensure different timestamps
                time.sleep(0.001)

            # Verify we have max caches
            assert len(convergence._similarity_cache_manager) == 5

            # Create one more - should evict oldest
            get_pairwise_similarity_cache("session_new")

            # Should still have max caches (not more)
            assert len(convergence._similarity_cache_manager) <= 5

            # New session should exist
            assert "session_new" in convergence._similarity_cache_manager

    def test_cleanup_removes_oldest_when_full(self):
        """Test that oldest cache is removed when at capacity."""
        from aragora.debate import convergence

        with patch.object(convergence, "MAX_SIMILARITY_CACHES", 3):
            # Create 3 caches with explicit timestamps
            sessions = ["oldest", "middle", "newest"]
            base_time = time.time()

            for i, session_id in enumerate(sessions):
                get_pairwise_similarity_cache(session_id)
                # Manually set timestamps to ensure ordering
                convergence._similarity_cache_timestamps[session_id] = base_time + i
                time.sleep(0.001)

            # Verify initial state
            assert len(convergence._similarity_cache_manager) == 3

            # Add new cache - should evict "oldest"
            get_pairwise_similarity_cache("brand_new")

            # "oldest" should be gone
            assert "oldest" not in convergence._similarity_cache_manager
            assert "brand_new" in convergence._similarity_cache_manager
            # Other caches should remain
            assert (
                "middle" in convergence._similarity_cache_manager
                or "newest" in convergence._similarity_cache_manager
            )


class TestCacheManagerConstants:
    """Tests for cache manager constants."""

    def test_max_similarity_caches_value(self):
        """Test that MAX_SIMILARITY_CACHES has expected value."""
        assert MAX_SIMILARITY_CACHES == 100

    def test_cache_manager_ttl_seconds_value(self):
        """Test that CACHE_MANAGER_TTL_SECONDS has expected value."""
        assert CACHE_MANAGER_TTL_SECONDS == 3600  # 1 hour


class TestPairwiseSimilarityCacheIntegration:
    """Integration tests for the full cache lifecycle."""

    def setup_method(self):
        """Clear global caches before each test."""
        from aragora.debate import convergence

        with convergence._similarity_cache_lock:
            for session_id in list(convergence._similarity_cache_manager.keys()):
                convergence._similarity_cache_manager[session_id].clear()
            convergence._similarity_cache_manager.clear()
            convergence._similarity_cache_timestamps.clear()

    def teardown_method(self):
        """Clear global caches after each test."""
        from aragora.debate import convergence

        with convergence._similarity_cache_lock:
            for session_id in list(convergence._similarity_cache_manager.keys()):
                convergence._similarity_cache_manager[session_id].clear()
            convergence._similarity_cache_manager.clear()
            convergence._similarity_cache_timestamps.clear()

    def test_full_lifecycle(self):
        """Test complete cache lifecycle: create, use, cleanup."""
        from aragora.debate import convergence

        session_id = "lifecycle_test"

        # Create cache
        cache = get_pairwise_similarity_cache(session_id)
        assert cache is not None
        assert session_id in convergence._similarity_cache_manager
        assert session_id in convergence._similarity_cache_timestamps

        # Use cache
        cache.put("text1", "text2", 0.85)
        result = cache.get("text1", "text2")
        assert result == 0.85

        # Cleanup
        cleanup_similarity_cache(session_id)
        assert session_id not in convergence._similarity_cache_manager
        assert session_id not in convergence._similarity_cache_timestamps

    def test_concurrent_cache_creation(self):
        """Test that concurrent cache creation is handled safely."""
        import threading

        from aragora.debate import convergence

        results = []
        errors = []

        def create_cache(session_id: str):
            try:
                cache = get_pairwise_similarity_cache(session_id)
                results.append((session_id, cache))
            except Exception as e:
                errors.append(e)

        # Create multiple caches concurrently
        threads = []
        for i in range(10):
            t = threading.Thread(target=create_cache, args=(f"concurrent_{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0

        # All caches should be created
        assert len(results) == 10

        # All should be in the manager
        for session_id, cache in results:
            assert session_id in convergence._similarity_cache_manager
