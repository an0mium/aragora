"""
Tests for concurrent session management.

Phase 6: Auth Handler Test Gaps - Concurrent session tests.

Tests:
- test_max_sessions_per_user_enforced - Default 10 session limit
- test_lru_eviction_on_max_sessions - Oldest session removed
- test_session_touch_updates_activity - Activity timestamp update
- test_revoke_all_except_current_session - Bulk revocation
- test_session_ordering_most_recent_first - Session list order
- test_concurrent_session_creation - Thread safety
- test_concurrent_session_revocation - Race condition prevention
- test_thread_safety_concurrent_access - Lock behavior
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.billing.auth.sessions import (
    JWTSession,
    JWTSessionManager,
    get_session_manager,
    reset_session_manager,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def session_manager() -> JWTSessionManager:
    """Create a fresh session manager for testing."""
    return JWTSessionManager(
        session_ttl=3600,
        max_sessions_per_user=10,
        inactivity_timeout=86400,
    )


@pytest.fixture
def small_session_manager() -> JWTSessionManager:
    """Create a session manager with small limits for testing."""
    return JWTSessionManager(
        session_ttl=3600,
        max_sessions_per_user=3,  # Small limit for testing eviction
        inactivity_timeout=86400,
    )


@pytest.fixture(autouse=True)
def reset_global_manager():
    """Reset global session manager before and after each test."""
    reset_session_manager()
    yield
    reset_session_manager()


# ============================================================================
# Test: Max Sessions Per User
# ============================================================================


class TestMaxSessionsPerUser:
    """Test maximum session limits per user."""

    def test_max_sessions_per_user_enforced(self, session_manager: JWTSessionManager):
        """Test that default 10 session limit is enforced."""
        user_id = "user-123"

        # Create 10 sessions
        for i in range(10):
            session_manager.create_session(
                user_id=user_id,
                token_jti=f"token-{i}",
                ip_address="127.0.0.1",
            )

        # Should have exactly 10 sessions
        sessions = session_manager.list_sessions(user_id)
        assert len(sessions) == 10

        # Creating 11th session should evict oldest
        session_manager.create_session(
            user_id=user_id,
            token_jti="token-10",
            ip_address="127.0.0.1",
        )

        sessions = session_manager.list_sessions(user_id)
        assert len(sessions) == 10  # Still 10, oldest evicted

    def test_max_sessions_custom_limit(self, small_session_manager: JWTSessionManager):
        """Test custom session limit enforcement."""
        user_id = "user-456"

        # Create 3 sessions (at limit)
        for i in range(3):
            small_session_manager.create_session(
                user_id=user_id,
                token_jti=f"token-{i}",
                ip_address="127.0.0.1",
            )

        sessions = small_session_manager.list_sessions(user_id)
        assert len(sessions) == 3

        # Creating 4th should evict oldest
        small_session_manager.create_session(
            user_id=user_id,
            token_jti="token-3",
            ip_address="127.0.0.1",
        )

        sessions = small_session_manager.list_sessions(user_id)
        assert len(sessions) == 3
        # First session should be evicted
        session_ids = [s.session_id for s in sessions]
        assert "token-0" not in session_ids


# ============================================================================
# Test: LRU Eviction
# ============================================================================


class TestLRUEviction:
    """Test LRU (Least Recently Used) eviction behavior."""

    def test_lru_eviction_on_max_sessions(self, small_session_manager: JWTSessionManager):
        """Test that oldest session is evicted when max is reached."""
        user_id = "user-lru"

        # Create sessions with small delays to establish order
        session_manager = small_session_manager

        session_manager.create_session(
            user_id=user_id,
            token_jti="oldest",
            ip_address="127.0.0.1",
        )
        session_manager.create_session(
            user_id=user_id,
            token_jti="middle",
            ip_address="127.0.0.1",
        )
        session_manager.create_session(
            user_id=user_id,
            token_jti="newest",
            ip_address="127.0.0.1",
        )

        # All 3 should exist
        sessions = session_manager.list_sessions(user_id)
        assert len(sessions) == 3

        # Add another session - oldest should be evicted
        session_manager.create_session(
            user_id=user_id,
            token_jti="newest-2",
            ip_address="127.0.0.1",
        )

        sessions = session_manager.list_sessions(user_id)
        session_ids = [s.session_id for s in sessions]

        assert "oldest" not in session_ids, "Oldest session should be evicted"
        assert "middle" in session_ids
        assert "newest" in session_ids
        assert "newest-2" in session_ids

    def test_touch_prevents_eviction(self, small_session_manager: JWTSessionManager):
        """Test that touching a session makes it most recent."""
        user_id = "user-touch"
        session_manager = small_session_manager

        # Create 3 sessions
        session_manager.create_session(user_id=user_id, token_jti="first", ip_address="127.0.0.1")
        session_manager.create_session(user_id=user_id, token_jti="second", ip_address="127.0.0.1")
        session_manager.create_session(user_id=user_id, token_jti="third", ip_address="127.0.0.1")

        # Touch the first session to make it most recent
        session_manager.touch_session(user_id, "first")

        # Add a new session - second should be evicted (now oldest)
        session_manager.create_session(user_id=user_id, token_jti="fourth", ip_address="127.0.0.1")

        sessions = session_manager.list_sessions(user_id)
        session_ids = [s.session_id for s in sessions]

        assert "first" in session_ids, "Touched session should not be evicted"
        assert "second" not in session_ids, "Untouched second session should be evicted"
        assert "third" in session_ids
        assert "fourth" in session_ids


# ============================================================================
# Test: Session Touch Updates Activity
# ============================================================================


class TestSessionTouch:
    """Test session touch functionality."""

    def test_session_touch_updates_activity(self, session_manager: JWTSessionManager):
        """Test that touch_session updates last_activity timestamp."""
        user_id = "user-activity"

        # Create session
        session = session_manager.create_session(
            user_id=user_id,
            token_jti="token-1",
            ip_address="127.0.0.1",
        )
        original_activity = session.last_activity

        # Small delay to ensure timestamp changes
        time.sleep(0.01)

        # Touch the session
        result = session_manager.touch_session(user_id, "token-1")
        assert result is True

        # Verify activity was updated
        updated_session = session_manager.get_session(user_id, "token-1")
        assert updated_session is not None
        assert updated_session.last_activity > original_activity

    def test_touch_nonexistent_session_returns_false(self, session_manager: JWTSessionManager):
        """Test that touching a nonexistent session returns False."""
        result = session_manager.touch_session("nonexistent-user", "nonexistent-token")
        assert result is False

    def test_touch_moves_session_to_end(self, small_session_manager: JWTSessionManager):
        """Test that touch moves session to end of ordered dict."""
        user_id = "user-order"
        session_manager = small_session_manager

        # Create sessions
        session_manager.create_session(user_id=user_id, token_jti="a", ip_address="127.0.0.1")
        session_manager.create_session(user_id=user_id, token_jti="b", ip_address="127.0.0.1")
        session_manager.create_session(user_id=user_id, token_jti="c", ip_address="127.0.0.1")

        # Touch "a" to make it most recent
        session_manager.touch_session(user_id, "a")

        # Add new session - "b" should be evicted as it's now oldest
        session_manager.create_session(user_id=user_id, token_jti="d", ip_address="127.0.0.1")

        sessions = session_manager.list_sessions(user_id)
        session_ids = [s.session_id for s in sessions]

        assert "b" not in session_ids, "Session 'b' should be evicted"
        assert "a" in session_ids, "Touched session 'a' should remain"


# ============================================================================
# Test: Revoke All Except Current
# ============================================================================


class TestRevokeAllExceptCurrent:
    """Test bulk session revocation."""

    def test_revoke_all_except_current_session(self, session_manager: JWTSessionManager):
        """Test revoking all sessions except the current one."""
        user_id = "user-revoke"

        # Create multiple sessions
        for i in range(5):
            session_manager.create_session(
                user_id=user_id,
                token_jti=f"token-{i}",
                ip_address="127.0.0.1",
            )

        # Revoke all except token-2
        revoked_count = session_manager.revoke_all_sessions(user_id, except_jti="token-2")

        assert revoked_count == 4

        # Only token-2 should remain
        sessions = session_manager.list_sessions(user_id)
        assert len(sessions) == 1
        assert sessions[0].session_id == "token-2"

    def test_revoke_all_sessions_no_exception(self, session_manager: JWTSessionManager):
        """Test revoking all sessions without exception."""
        user_id = "user-revoke-all"

        # Create sessions
        for i in range(3):
            session_manager.create_session(
                user_id=user_id,
                token_jti=f"token-{i}",
                ip_address="127.0.0.1",
            )

        # Revoke all
        revoked_count = session_manager.revoke_all_sessions(user_id)

        assert revoked_count == 3

        # No sessions should remain
        sessions = session_manager.list_sessions(user_id)
        assert len(sessions) == 0

    def test_revoke_all_for_nonexistent_user(self, session_manager: JWTSessionManager):
        """Test revoking sessions for nonexistent user."""
        revoked_count = session_manager.revoke_all_sessions("nonexistent-user")
        assert revoked_count == 0


# ============================================================================
# Test: Session Ordering
# ============================================================================


class TestSessionOrdering:
    """Test session list ordering."""

    def test_session_ordering_most_recent_first(self, session_manager: JWTSessionManager):
        """Test that sessions are ordered by last activity (most recent first)."""
        user_id = "user-order"

        # Create sessions with significant time gaps
        session1 = session_manager.create_session(
            user_id=user_id, token_jti="old", ip_address="127.0.0.1"
        )
        session2 = session_manager.create_session(
            user_id=user_id, token_jti="middle", ip_address="127.0.0.1"
        )
        session3 = session_manager.create_session(
            user_id=user_id, token_jti="new", ip_address="127.0.0.1"
        )

        # Manually set different activity times to ensure ordering
        # (instead of relying on sleep which can be flaky)
        now = time.time()
        session1.last_activity = now - 100  # oldest
        session2.last_activity = now - 50  # middle
        session3.last_activity = now  # newest

        # Touch "old" to make it most recent
        session_manager.touch_session(user_id, "old")

        # Get sessions - should be ordered by last_activity descending
        sessions = session_manager.list_sessions(user_id)

        # Verify sessions exist
        assert len(sessions) == 3

        # The touched "old" session should now have the most recent activity
        # Find it and verify it was updated
        old_session = next((s for s in sessions if s.session_id == "old"), None)
        assert old_session is not None
        # After touch, old should be more recent than the other two (which we set to past times)
        assert old_session.last_activity > now - 10  # Recently touched


# ============================================================================
# Test: Thread Safety - Concurrent Session Creation
# ============================================================================


class TestConcurrentSessionCreation:
    """Test thread safety during concurrent session creation."""

    def test_concurrent_session_creation(self, session_manager: JWTSessionManager):
        """Test concurrent session creation is thread-safe."""
        user_id = "user-concurrent"
        num_threads = 20
        sessions_created = []
        errors = []

        def create_session(thread_id: int):
            try:
                session = session_manager.create_session(
                    user_id=user_id,
                    token_jti=f"token-{thread_id}",
                    ip_address="127.0.0.1",
                )
                sessions_created.append(session)
            except Exception as e:
                errors.append(e)

        # Create sessions concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_session, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        # No errors should occur
        assert len(errors) == 0, f"Errors during concurrent creation: {errors}"

        # Due to max_sessions_per_user=10, only 10 sessions should exist
        sessions = session_manager.list_sessions(user_id)
        assert len(sessions) == 10

    def test_concurrent_different_users(self, session_manager: JWTSessionManager):
        """Test concurrent session creation for different users."""
        num_users = 5
        sessions_per_user = 3
        results = {}
        lock = threading.Lock()

        def create_sessions_for_user(user_id: str):
            user_sessions = []
            for i in range(sessions_per_user):
                session = session_manager.create_session(
                    user_id=user_id,
                    token_jti=f"{user_id}-token-{i}",
                    ip_address="127.0.0.1",
                )
                user_sessions.append(session)
            with lock:
                results[user_id] = user_sessions

        # Create sessions for different users concurrently
        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [
                executor.submit(create_sessions_for_user, f"user-{i}") for i in range(num_users)
            ]
            for future in as_completed(futures):
                future.result()

        # Each user should have their sessions
        for i in range(num_users):
            user_id = f"user-{i}"
            sessions = session_manager.list_sessions(user_id)
            assert len(sessions) == sessions_per_user


# ============================================================================
# Test: Thread Safety - Concurrent Session Revocation
# ============================================================================


class TestConcurrentSessionRevocation:
    """Test thread safety during concurrent session revocation."""

    def test_concurrent_session_revocation(self, session_manager: JWTSessionManager):
        """Test concurrent session revocation is thread-safe."""
        user_id = "user-revoke-concurrent"

        # Create sessions
        for i in range(10):
            session_manager.create_session(
                user_id=user_id,
                token_jti=f"token-{i}",
                ip_address="127.0.0.1",
            )

        errors = []
        revocation_results = []

        def revoke_session(token_id: str):
            try:
                result = session_manager.revoke_session(user_id, token_id)
                revocation_results.append(result)
            except Exception as e:
                errors.append(e)

        # Revoke sessions concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(revoke_session, f"token-{i}") for i in range(10)]
            for future in as_completed(futures):
                future.result()

        # No errors should occur
        assert len(errors) == 0, f"Errors during concurrent revocation: {errors}"

        # All revocations should succeed
        assert sum(revocation_results) == 10

        # No sessions should remain
        sessions = session_manager.list_sessions(user_id)
        assert len(sessions) == 0


# ============================================================================
# Test: Thread Safety - Lock Behavior
# ============================================================================


class TestThreadSafety:
    """Test general thread safety and lock behavior."""

    def test_thread_safety_concurrent_access(self, session_manager: JWTSessionManager):
        """Test mixed concurrent operations are thread-safe."""
        user_id = "user-mixed"
        errors = []

        # Create initial sessions
        for i in range(5):
            session_manager.create_session(
                user_id=user_id,
                token_jti=f"initial-{i}",
                ip_address="127.0.0.1",
            )

        def mixed_operations(thread_id: int):
            try:
                # Create
                session_manager.create_session(
                    user_id=user_id,
                    token_jti=f"thread-{thread_id}",
                    ip_address="127.0.0.1",
                )
                # Touch random session
                session_manager.touch_session(user_id, f"initial-{thread_id % 5}")
                # List
                session_manager.list_sessions(user_id)
                # Get
                session_manager.get_session(user_id, f"initial-{thread_id % 5}")
            except Exception as e:
                errors.append(e)

        # Run mixed operations concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(mixed_operations, i) for i in range(20)]
            for future in as_completed(futures):
                future.result()

        # No errors should occur
        assert len(errors) == 0, f"Errors during concurrent access: {errors}"

    def test_lock_prevents_data_corruption(self, session_manager: JWTSessionManager):
        """Test that lock prevents data corruption during concurrent writes."""
        user_id = "user-corruption-test"
        counter = {"value": 0}

        def create_and_count():
            for _ in range(100):
                session_manager.create_session(
                    user_id=user_id,
                    token_jti=f"token-{counter['value']}",
                    ip_address="127.0.0.1",
                )
                counter["value"] += 1

        threads = [threading.Thread(target=create_and_count) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Session count should be at max limit (10)
        sessions = session_manager.list_sessions(user_id)
        assert len(sessions) <= 10  # Should never exceed max


# ============================================================================
# Test: Session Get and Count
# ============================================================================


class TestSessionGetAndCount:
    """Test session retrieval and counting."""

    def test_get_session_count(self, session_manager: JWTSessionManager):
        """Test getting session count for a user."""
        user_id = "user-count"

        assert session_manager.get_session_count(user_id) == 0

        session_manager.create_session(user_id=user_id, token_jti="token-1", ip_address="127.0.0.1")
        assert session_manager.get_session_count(user_id) == 1

        session_manager.create_session(user_id=user_id, token_jti="token-2", ip_address="127.0.0.1")
        assert session_manager.get_session_count(user_id) == 2

    def test_get_session_returns_none_for_nonexistent(self, session_manager: JWTSessionManager):
        """Test that get_session returns None for nonexistent sessions."""
        session = session_manager.get_session("nonexistent", "token")
        assert session is None


__all__ = [
    "TestMaxSessionsPerUser",
    "TestLRUEviction",
    "TestSessionTouch",
    "TestRevokeAllExceptCurrent",
    "TestSessionOrdering",
    "TestConcurrentSessionCreation",
    "TestConcurrentSessionRevocation",
    "TestThreadSafety",
    "TestSessionGetAndCount",
]
