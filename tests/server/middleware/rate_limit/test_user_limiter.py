"""
Tests for Per-User Rate Limiter.

Covers:
- UserRateLimiter initialization and configuration
- Per-user, per-action rate limiting
- Action-specific rate limits (debate_create, vote, export, etc.)
- User quota management and exhaustion
- LRU eviction of stale entries
- Fallback to IP-based limiting for unauthenticated users
- Cleanup functionality
- User status and statistics
- Thread safety
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def clean_user_limiter():
    """Reset global user limiter before and after test."""
    from aragora.server.middleware.rate_limit.user_limiter import (
        get_user_rate_limiter,
    )

    limiter = get_user_rate_limiter()
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture
def fresh_user_limiter():
    """Create a fresh UserRateLimiter instance."""
    from aragora.server.middleware.rate_limit.user_limiter import UserRateLimiter

    return UserRateLimiter()


# -----------------------------------------------------------------------------
# UserRateLimiter Initialization Tests
# -----------------------------------------------------------------------------


class TestUserRateLimiterInitialization:
    """Tests for UserRateLimiter initialization."""

    def test_default_initialization(self):
        """Should initialize with default values."""
        from aragora.server.middleware.rate_limit.user_limiter import (
            USER_RATE_LIMITS,
            UserRateLimiter,
        )

        limiter = UserRateLimiter()

        assert limiter.action_limits == USER_RATE_LIMITS
        assert limiter.default_limit == 60
        assert limiter.max_users == 10000

    def test_custom_action_limits(self):
        """Should accept custom action limits."""
        from aragora.server.middleware.rate_limit.user_limiter import UserRateLimiter

        custom_limits = {"custom_action": 100, "another": 50}
        limiter = UserRateLimiter(action_limits=custom_limits)

        assert limiter.action_limits == custom_limits

    def test_custom_default_limit(self):
        """Should accept custom default limit."""
        from aragora.server.middleware.rate_limit.user_limiter import UserRateLimiter

        limiter = UserRateLimiter(default_limit=100)

        assert limiter.default_limit == 100

    def test_custom_max_users(self):
        """Should accept custom max_users."""
        from aragora.server.middleware.rate_limit.user_limiter import UserRateLimiter

        limiter = UserRateLimiter(max_users=5000)

        assert limiter.max_users == 5000


# -----------------------------------------------------------------------------
# Action Limit Configuration Tests
# -----------------------------------------------------------------------------


class TestActionLimitConfiguration:
    """Tests for action-specific rate limit configuration."""

    def test_get_action_limit_known_action(self, fresh_user_limiter):
        """Should return correct limit for known action."""
        limit = fresh_user_limiter.get_action_limit("debate_create")
        assert limit == 10

    def test_get_action_limit_unknown_action(self, fresh_user_limiter):
        """Should return default limit for unknown action."""
        limit = fresh_user_limiter.get_action_limit("unknown_action")
        assert limit == 60

    def test_default_user_rate_limits(self):
        """Should have expected default action limits."""
        from aragora.server.middleware.rate_limit.user_limiter import USER_RATE_LIMITS

        assert USER_RATE_LIMITS["default"] == 60
        assert USER_RATE_LIMITS["debate_create"] == 10
        assert USER_RATE_LIMITS["debate_search"] == 30
        assert USER_RATE_LIMITS["vote"] == 30
        assert USER_RATE_LIMITS["agent_call"] == 120
        assert USER_RATE_LIMITS["export"] == 5
        assert USER_RATE_LIMITS["admin"] == 300
        assert USER_RATE_LIMITS["batch_submit"] == 2
        assert USER_RATE_LIMITS["evidence_collect"] == 10
        assert USER_RATE_LIMITS["knowledge_query"] == 20
        assert USER_RATE_LIMITS["memory_cleanup"] == 5
        assert USER_RATE_LIMITS["plugin_install"] == 3
        assert USER_RATE_LIMITS["slack_command"] == 10
        assert USER_RATE_LIMITS["slack_debate"] == 5
        assert USER_RATE_LIMITS["slack_gauntlet"] == 3


# -----------------------------------------------------------------------------
# Per-User Rate Limiting Tests
# -----------------------------------------------------------------------------


class TestPerUserRateLimiting:
    """Tests for per-user rate limit enforcement."""

    def test_allows_first_request(self, fresh_user_limiter):
        """First request should always be allowed."""
        result = fresh_user_limiter.allow("user-123", action="default")

        assert result.allowed is True
        assert result.key == "user:user-123:default"

    def test_allows_requests_under_limit(self, fresh_user_limiter):
        """Requests under limit should be allowed."""
        for _ in range(10):
            result = fresh_user_limiter.allow("user-123", action="default")
            assert result.allowed is True

    def test_returns_remaining_count(self, fresh_user_limiter):
        """Should return remaining token count."""
        result = fresh_user_limiter.allow("user-123", action="default")

        assert result.remaining >= 0
        assert result.limit == 60  # Default action limit

    def test_different_users_independent(self, fresh_user_limiter):
        """Different users should have independent limits."""
        # Create custom limiter with low limit
        from aragora.server.middleware.rate_limit.user_limiter import UserRateLimiter

        limiter = UserRateLimiter(action_limits={"test": 2})

        # Exhaust limit for user-1
        for _ in range(10):
            limiter.allow("user-1", action="test")

        # user-2 should still work
        result = limiter.allow("user-2", action="test")
        assert result.allowed is True


# -----------------------------------------------------------------------------
# Per-Action Rate Limiting Tests
# -----------------------------------------------------------------------------


class TestPerActionRateLimiting:
    """Tests for per-action rate limit enforcement."""

    def test_different_actions_independent(self, fresh_user_limiter):
        """Different actions should have independent limits."""
        # Make many debate_create requests
        for _ in range(20):
            fresh_user_limiter.allow("user-123", action="debate_create")

        # vote action should still work
        result = fresh_user_limiter.allow("user-123", action="vote")
        assert result.allowed is True

    def test_action_specific_limits_enforced(self):
        """Action-specific limits should be enforced."""
        from aragora.server.middleware.rate_limit.user_limiter import UserRateLimiter

        limiter = UserRateLimiter(action_limits={"expensive": 2}, default_limit=100)

        # Should exhaust expensive action quickly
        allowed = 0
        for _ in range(10):
            result = limiter.allow("user-123", action="expensive")
            if result.allowed:
                allowed += 1

        # Burst = 2 * 2 = 4, so should allow around 4
        assert allowed < 10

    def test_export_has_low_limit(self, fresh_user_limiter):
        """Export action should have low limit (5 rpm)."""
        limit = fresh_user_limiter.get_action_limit("export")
        assert limit == 5

    def test_admin_has_high_limit(self, fresh_user_limiter):
        """Admin action should have high limit (300 rpm)."""
        limit = fresh_user_limiter.get_action_limit("admin")
        assert limit == 300


# -----------------------------------------------------------------------------
# Quota Exhaustion Tests
# -----------------------------------------------------------------------------


class TestQuotaExhaustion:
    """Tests for quota exhaustion handling."""

    def test_denies_after_quota_exhausted(self):
        """Should deny after quota exhausted."""
        from aragora.server.middleware.rate_limit.user_limiter import UserRateLimiter

        limiter = UserRateLimiter(action_limits={"limited": 2})

        # Exhaust quota (burst = 2 * 2 = 4)
        for _ in range(10):
            limiter.allow("user-123", action="limited")

        result = limiter.allow("user-123", action="limited")
        assert result.allowed is False

    def test_returns_retry_after_when_exhausted(self):
        """Should return retry_after when quota exhausted."""
        from aragora.server.middleware.rate_limit.user_limiter import UserRateLimiter

        limiter = UserRateLimiter(action_limits={"limited": 1})

        # Exhaust quota
        for _ in range(10):
            limiter.allow("user-123", action="limited")

        result = limiter.allow("user-123", action="limited")

        if not result.allowed:
            assert result.retry_after > 0

    def test_quota_refills_over_time(self):
        """Quota should refill over time."""
        from aragora.server.middleware.rate_limit.user_limiter import UserRateLimiter

        limiter = UserRateLimiter(action_limits={"refill": 60})  # 1 per second

        # Consume some tokens
        for _ in range(5):
            limiter.allow("user-123", action="refill")

        # Get the bucket and simulate time passing
        bucket = limiter._user_buckets.get("refill", {}).get("user-123")
        if bucket:
            # Simulate 2 seconds passing
            bucket.last_refill = time.monotonic() - 2

        result = limiter.allow("user-123", action="refill")
        assert result.allowed is True


# -----------------------------------------------------------------------------
# LRU Eviction Tests
# -----------------------------------------------------------------------------


class TestLRUEviction:
    """Tests for LRU eviction of stale entries."""

    def test_evicts_old_entries_when_max_reached(self):
        """Should evict old entries when max_users reached."""
        from aragora.server.middleware.rate_limit.user_limiter import UserRateLimiter

        limiter = UserRateLimiter(max_users=10)

        # Add many users for same action
        for i in range(20):
            limiter.allow(f"user-{i}", action="test")

        # Should have fewer entries than added due to eviction
        buckets = limiter._user_buckets.get("test", {})
        assert len(buckets) <= 10

    def test_moves_recently_used_to_end(self):
        """Recently used entries should move to end (LRU)."""
        from aragora.server.middleware.rate_limit.user_limiter import UserRateLimiter

        limiter = UserRateLimiter(max_users=1000)

        # Add users in order
        for i in range(5):
            limiter.allow(f"user-{i}", action="test")

        # Access user-0 again
        limiter.allow("user-0", action="test")

        # user-0 should be at end now
        buckets = limiter._user_buckets.get("test", {})
        keys = list(buckets.keys())
        assert keys[-1] == "user-0"


# -----------------------------------------------------------------------------
# Cleanup Tests
# -----------------------------------------------------------------------------


class TestCleanup:
    """Tests for cleanup functionality."""

    def test_cleanup_removes_stale_entries(self):
        """cleanup() should remove stale entries."""
        from aragora.server.middleware.rate_limit.user_limiter import UserRateLimiter

        limiter = UserRateLimiter()

        # Add a user
        limiter.allow("user-123", action="test")

        # Get bucket and make it stale
        bucket = limiter._user_buckets.get("test", {}).get("user-123")
        if bucket:
            bucket.last_refill = time.monotonic() - 700  # > 600 seconds

        removed = limiter.cleanup(max_age_seconds=600)

        assert removed >= 1

    def test_cleanup_removes_empty_action_buckets(self):
        """cleanup() should remove empty action bucket dicts."""
        from aragora.server.middleware.rate_limit.user_limiter import UserRateLimiter

        limiter = UserRateLimiter()

        # Add a user
        limiter.allow("user-123", action="test")

        # Get bucket and make it stale
        bucket = limiter._user_buckets.get("test", {}).get("user-123")
        if bucket:
            bucket.last_refill = time.monotonic() - 700

        limiter.cleanup(max_age_seconds=600)

        # Empty action dict should be removed
        assert "test" not in limiter._user_buckets


# -----------------------------------------------------------------------------
# User Status Tests
# -----------------------------------------------------------------------------


class TestUserStatus:
    """Tests for get_user_status functionality."""

    def test_get_user_status_empty(self, fresh_user_limiter):
        """Should return empty status for unknown user."""
        status = fresh_user_limiter.get_user_status("unknown-user")
        assert status == {}

    def test_get_user_status_with_activity(self, fresh_user_limiter):
        """Should return status for active user."""
        fresh_user_limiter.allow("user-123", action="vote")
        fresh_user_limiter.allow("user-123", action="debate_create")

        status = fresh_user_limiter.get_user_status("user-123")

        assert "vote" in status
        assert "debate_create" in status
        assert "remaining" in status["vote"]
        assert "limit" in status["vote"]
        assert "retry_after" in status["vote"]


# -----------------------------------------------------------------------------
# Statistics Tests
# -----------------------------------------------------------------------------


class TestStatistics:
    """Tests for get_stats functionality."""

    def test_get_stats_empty(self, fresh_user_limiter):
        """Should return stats for empty limiter."""
        stats = fresh_user_limiter.get_stats()

        assert "action_buckets" in stats
        assert "action_limits" in stats
        assert "total_users" in stats
        assert stats["total_users"] == 0

    def test_get_stats_with_activity(self, fresh_user_limiter):
        """Should return accurate stats with activity."""
        fresh_user_limiter.allow("user-1", action="vote")
        fresh_user_limiter.allow("user-2", action="vote")
        fresh_user_limiter.allow("user-1", action="export")

        stats = fresh_user_limiter.get_stats()

        assert stats["action_buckets"]["vote"] == 2
        assert stats["action_buckets"]["export"] == 1
        assert stats["total_users"] == 3  # 2 + 1


# -----------------------------------------------------------------------------
# Reset Tests
# -----------------------------------------------------------------------------


class TestReset:
    """Tests for reset functionality."""

    def test_reset_clears_all_state(self, fresh_user_limiter):
        """reset() should clear all state."""
        fresh_user_limiter.allow("user-1", action="vote")
        fresh_user_limiter.allow("user-2", action="export")

        fresh_user_limiter.reset()

        stats = fresh_user_limiter.get_stats()
        assert stats["total_users"] == 0


# -----------------------------------------------------------------------------
# Global Limiter Tests
# -----------------------------------------------------------------------------


class TestGlobalLimiter:
    """Tests for global user rate limiter instance."""

    def test_get_user_rate_limiter_singleton(self, clean_user_limiter):
        """get_user_rate_limiter should return singleton."""
        from aragora.server.middleware.rate_limit.user_limiter import (
            get_user_rate_limiter,
        )

        limiter1 = get_user_rate_limiter()
        limiter2 = get_user_rate_limiter()

        assert limiter1 is limiter2


# -----------------------------------------------------------------------------
# check_user_rate_limit Helper Tests
# -----------------------------------------------------------------------------


class TestCheckUserRateLimit:
    """Tests for check_user_rate_limit helper function."""

    def test_check_with_handler_client_address(self, clean_user_limiter):
        """Should extract IP from handler.client_address."""
        from aragora.server.middleware.rate_limit.user_limiter import (
            check_user_rate_limit,
        )

        handler = MagicMock()
        handler.client_address = ("192.168.1.100", 12345)
        handler.headers = MagicMock()
        handler.headers.get = MagicMock(return_value="")

        result = check_user_rate_limit(handler, action="default")

        assert result.allowed is True
        assert "ip:192.168.1.100" in result.key

    def test_check_with_x_forwarded_for(self, clean_user_limiter):
        """Should respect X-Forwarded-For from trusted proxy."""
        from aragora.server.middleware.rate_limit.user_limiter import (
            check_user_rate_limit,
        )

        handler = MagicMock()
        handler.client_address = ("127.0.0.1", 12345)  # Trusted proxy
        handler.headers = MagicMock()
        handler.headers.get = MagicMock(
            side_effect=lambda k, d="": ("10.0.0.50" if k == "X-Forwarded-For" else d)
        )

        result = check_user_rate_limit(handler, action="default")

        assert result.allowed is True

    def test_check_with_authenticated_user(self, clean_user_limiter):
        """Should use user_id for authenticated users."""
        from aragora.server.middleware.rate_limit.user_limiter import (
            check_user_rate_limit,
        )

        handler = MagicMock()
        handler.client_address = ("192.168.1.100", 12345)
        handler.headers = MagicMock()
        handler.headers.get = MagicMock(return_value="")

        # Mock user store and auth extraction
        user_store = MagicMock()

        # Patch at the actual import location in billing.jwt_auth
        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_auth = MagicMock()
            mock_auth.is_authenticated = True
            mock_auth.user_id = "authenticated-user-456"
            mock_extract.return_value = mock_auth

            result = check_user_rate_limit(handler, user_store, action="vote")

            assert result.allowed is True
            assert "authenticated-user-456" in result.key

    def test_check_fallback_when_extraction_fails(self, clean_user_limiter):
        """Should fallback to IP when auth extraction fails."""
        from aragora.server.middleware.rate_limit.user_limiter import (
            check_user_rate_limit,
        )

        handler = MagicMock()
        handler.client_address = ("192.168.1.100", 12345)
        handler.headers = MagicMock()
        handler.headers.get = MagicMock(return_value="")

        user_store = MagicMock()

        # Patch at the actual import location in billing.jwt_auth
        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.side_effect = Exception("Auth failed")

            result = check_user_rate_limit(handler, user_store, action="default")

            assert result.allowed is True
            assert "ip:192.168.1.100" in result.key


# -----------------------------------------------------------------------------
# Thread Safety Tests
# -----------------------------------------------------------------------------


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_requests_thread_safe(self, fresh_user_limiter):
        """Concurrent requests should be thread-safe."""
        results = []
        errors = []

        def make_requests():
            try:
                for _ in range(50):
                    result = fresh_user_limiter.allow("user-shared", action="default")
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=make_requests) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 250

    def test_concurrent_different_users(self, fresh_user_limiter):
        """Concurrent requests from different users should work."""
        results_per_user = {}
        lock = threading.Lock()

        def make_requests(user_id):
            results = []
            for _ in range(10):
                result = fresh_user_limiter.allow(user_id, action="default")
                results.append(result)
            with lock:
                results_per_user[user_id] = results

        threads = [threading.Thread(target=make_requests, args=(f"user-{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each user should have all requests allowed (under limit)
        for user_id, results in results_per_user.items():
            allowed = sum(1 for r in results if r.allowed)
            assert allowed == 10


# -----------------------------------------------------------------------------
# Edge Cases Tests
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_user_id(self, fresh_user_limiter):
        """Should handle empty user_id."""
        result = fresh_user_limiter.allow("", action="default")
        assert result.allowed is True

    def test_empty_action(self, fresh_user_limiter):
        """Should handle empty action (uses default)."""
        # Empty string gets default limit
        from aragora.server.middleware.rate_limit.user_limiter import UserRateLimiter

        limiter = UserRateLimiter(action_limits={})
        result = limiter.allow("user-123", action="")
        assert result.limit == 60  # default_limit

    def test_very_long_user_id(self, fresh_user_limiter):
        """Should handle very long user_id."""
        long_user_id = "user-" + "a" * 10000
        result = fresh_user_limiter.allow(long_user_id, action="default")
        assert result.allowed is True

    def test_special_characters_in_user_id(self, fresh_user_limiter):
        """Should handle special characters in user_id."""
        result = fresh_user_limiter.allow("user:with:colons@special.chars", action="default")
        assert result.allowed is True

    def test_handler_without_client_address(self, clean_user_limiter):
        """Should handle handler without client_address."""
        from aragora.server.middleware.rate_limit.user_limiter import (
            check_user_rate_limit,
        )

        handler = MagicMock(spec=[])  # No attributes

        result = check_user_rate_limit(handler, action="default")

        assert result.allowed is True
        assert "anon" in result.key

    def test_handler_with_none_client_address(self, clean_user_limiter):
        """Should handle handler with None client_address."""
        from aragora.server.middleware.rate_limit.user_limiter import (
            check_user_rate_limit,
        )

        handler = MagicMock()
        handler.client_address = None
        handler.headers = MagicMock()
        handler.headers.get = MagicMock(return_value="")

        result = check_user_rate_limit(handler, action="default")

        assert result.allowed is True
