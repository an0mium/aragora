"""
Tests for Tier-based Rate Limiter.

Covers:
- TierRateLimiter initialization and configuration
- Tier-specific rate limits (free, starter, professional, enterprise)
- Tier isolation (different tiers don't share buckets)
- Quota management per tier
- LRU eviction
- Fallback to free tier for unknown tiers
- Global limiter and helper functions
- Thread safety
- Configuration validation
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
def clean_tier_limiter():
    """Reset global tier limiter before and after test."""
    from aragora.server.middleware.rate_limit.tier_limiter import (
        get_tier_rate_limiter,
    )

    limiter = get_tier_rate_limiter()
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture
def fresh_tier_limiter():
    """Create a fresh TierRateLimiter instance."""
    from aragora.server.middleware.rate_limit.tier_limiter import TierRateLimiter

    return TierRateLimiter()


# -----------------------------------------------------------------------------
# TierRateLimiter Initialization Tests
# -----------------------------------------------------------------------------


class TestTierRateLimiterInitialization:
    """Tests for TierRateLimiter initialization."""

    def test_default_initialization(self):
        """Should initialize with default tier limits."""
        from aragora.server.middleware.rate_limit.tier_limiter import (
            TIER_RATE_LIMITS,
            TierRateLimiter,
        )

        limiter = TierRateLimiter()

        assert limiter.tier_limits == TIER_RATE_LIMITS
        assert limiter.max_entries == 10000

    def test_custom_tier_limits(self):
        """Should accept custom tier limits."""
        from aragora.server.middleware.rate_limit.tier_limiter import TierRateLimiter

        custom_limits = {
            "bronze": (20, 40),
            "silver": (50, 100),
            "gold": (200, 400),
        }
        limiter = TierRateLimiter(tier_limits=custom_limits)

        assert limiter.tier_limits == custom_limits

    def test_custom_max_entries(self):
        """Should accept custom max_entries."""
        from aragora.server.middleware.rate_limit.tier_limiter import TierRateLimiter

        limiter = TierRateLimiter(max_entries=5000)

        assert limiter.max_entries == 5000

    def test_creates_tier_buckets_on_init(self):
        """Should create empty bucket dicts for each tier on init."""
        from aragora.server.middleware.rate_limit.tier_limiter import TierRateLimiter

        limiter = TierRateLimiter()

        assert "free" in limiter._tier_buckets
        assert "starter" in limiter._tier_buckets
        assert "professional" in limiter._tier_buckets
        assert "enterprise" in limiter._tier_buckets


# -----------------------------------------------------------------------------
# Default Tier Configuration Tests
# -----------------------------------------------------------------------------


class TestDefaultTierConfiguration:
    """Tests for default tier rate limit configuration."""

    def test_free_tier_limits(self):
        """Free tier should have lowest limits."""
        from aragora.server.middleware.rate_limit.tier_limiter import TIER_RATE_LIMITS

        rate, burst = TIER_RATE_LIMITS["free"]
        assert rate == 10
        assert burst == 60

    def test_starter_tier_limits(self):
        """Starter tier should have moderate limits."""
        from aragora.server.middleware.rate_limit.tier_limiter import TIER_RATE_LIMITS

        rate, burst = TIER_RATE_LIMITS["starter"]
        assert rate == 50
        assert burst == 100

    def test_professional_tier_limits(self):
        """Professional tier should have higher limits."""
        from aragora.server.middleware.rate_limit.tier_limiter import TIER_RATE_LIMITS

        rate, burst = TIER_RATE_LIMITS["professional"]
        assert rate == 200
        assert burst == 400

    def test_enterprise_tier_limits(self):
        """Enterprise tier should have highest limits."""
        from aragora.server.middleware.rate_limit.tier_limiter import TIER_RATE_LIMITS

        rate, burst = TIER_RATE_LIMITS["enterprise"]
        assert rate == 1000
        assert burst == 2000


# -----------------------------------------------------------------------------
# get_tier_limits Tests
# -----------------------------------------------------------------------------


class TestGetTierLimits:
    """Tests for get_tier_limits method."""

    def test_get_known_tier_limits(self, fresh_tier_limiter):
        """Should return correct limits for known tiers."""
        rate, burst = fresh_tier_limiter.get_tier_limits("professional")
        assert rate == 200
        assert burst == 400

    def test_get_unknown_tier_defaults_to_free(self, fresh_tier_limiter):
        """Unknown tier should default to free tier limits."""
        rate, burst = fresh_tier_limiter.get_tier_limits("unknown_tier")
        assert rate == 10
        assert burst == 60

    def test_get_tier_limits_case_insensitive(self, fresh_tier_limiter):
        """Tier lookup should be case-insensitive."""
        rate1, burst1 = fresh_tier_limiter.get_tier_limits("ENTERPRISE")
        rate2, burst2 = fresh_tier_limiter.get_tier_limits("enterprise")

        assert rate1 == rate2
        assert burst1 == burst2


# -----------------------------------------------------------------------------
# Tier-Specific Rate Limiting Tests
# -----------------------------------------------------------------------------


class TestTierSpecificRateLimiting:
    """Tests for tier-specific rate limit enforcement."""

    def test_free_tier_allows_first_request(self, fresh_tier_limiter):
        """Free tier should allow first request."""
        result = fresh_tier_limiter.allow("user-123", tier="free")

        assert result.allowed is True
        assert result.limit == 10
        assert result.key == "tier:free:user-123"

    def test_enterprise_tier_allows_many_requests(self, fresh_tier_limiter):
        """Enterprise tier should allow many requests."""
        for _ in range(100):
            result = fresh_tier_limiter.allow("user-enterprise", tier="enterprise")
            assert result.allowed is True

    def test_free_tier_denies_after_burst_exhausted(self):
        """Free tier should deny after burst exhausted."""
        from aragora.server.middleware.rate_limit.tier_limiter import TierRateLimiter

        limiter = TierRateLimiter(
            tier_limits={"free": (2, 4)}  # Very low for testing
        )

        # Exhaust burst (4 tokens)
        for _ in range(10):
            limiter.allow("user-123", tier="free")

        result = limiter.allow("user-123", tier="free")
        assert result.allowed is False

    def test_returns_remaining_count(self, fresh_tier_limiter):
        """Should return remaining token count."""
        result = fresh_tier_limiter.allow("user-123", tier="starter")

        assert result.remaining >= 0
        assert result.limit == 50  # Starter tier rate


# -----------------------------------------------------------------------------
# Tier Isolation Tests
# -----------------------------------------------------------------------------


class TestTierIsolation:
    """Tests for tier isolation (different tiers don't share buckets)."""

    def test_same_user_different_tiers_independent(self, fresh_tier_limiter):
        """Same user in different tiers should have independent limits."""
        # Exhaust free tier for user
        for _ in range(100):
            fresh_tier_limiter.allow("user-123", tier="free")

        # Same user on professional tier should work
        result = fresh_tier_limiter.allow("user-123", tier="professional")
        assert result.allowed is True

    def test_different_users_same_tier_independent(self, fresh_tier_limiter):
        """Different users in same tier should have independent limits."""
        from aragora.server.middleware.rate_limit.tier_limiter import TierRateLimiter

        limiter = TierRateLimiter(tier_limits={"test": (2, 4)})

        # Exhaust user-1
        for _ in range(10):
            limiter.allow("user-1", tier="test")

        # user-2 should still work
        result = limiter.allow("user-2", tier="test")
        assert result.allowed is True

    def test_tier_buckets_track_separately(self, fresh_tier_limiter):
        """Each tier should track its own bucket set."""
        fresh_tier_limiter.allow("user-1", tier="free")
        fresh_tier_limiter.allow("user-1", tier="starter")
        fresh_tier_limiter.allow("user-2", tier="free")

        stats = fresh_tier_limiter.get_stats()

        assert stats["tier_buckets"]["free"] == 2  # user-1, user-2
        assert stats["tier_buckets"]["starter"] == 1  # user-1


# -----------------------------------------------------------------------------
# LRU Eviction Tests
# -----------------------------------------------------------------------------


class TestLRUEviction:
    """Tests for LRU eviction of stale entries."""

    def test_evicts_old_entries_when_max_reached(self):
        """Should evict old entries when max_entries reached per tier."""
        from aragora.server.middleware.rate_limit.tier_limiter import TierRateLimiter

        limiter = TierRateLimiter(max_entries=20)  # 5 per tier (20/4)

        # Add many users to free tier
        for i in range(15):
            limiter.allow(f"user-{i}", tier="free")

        # Should have evicted some
        buckets = limiter._tier_buckets.get("free", {})
        assert len(buckets) <= 5

    def test_moves_recently_used_to_end(self):
        """Recently used entries should move to end (LRU)."""
        from aragora.server.middleware.rate_limit.tier_limiter import TierRateLimiter

        limiter = TierRateLimiter(max_entries=1000)

        # Add users in order
        for i in range(5):
            limiter.allow(f"user-{i}", tier="free")

        # Access user-0 again
        limiter.allow("user-0", tier="free")

        # user-0 should be at end now
        buckets = limiter._tier_buckets.get("free", {})
        keys = list(buckets.keys())
        assert keys[-1] == "user-0"


# -----------------------------------------------------------------------------
# Statistics Tests
# -----------------------------------------------------------------------------


class TestStatistics:
    """Tests for get_stats functionality."""

    def test_get_stats_empty(self, fresh_tier_limiter):
        """Should return stats for empty limiter."""
        stats = fresh_tier_limiter.get_stats()

        assert "tier_buckets" in stats
        assert "tier_limits" in stats
        assert stats["tier_buckets"]["free"] == 0
        assert stats["tier_buckets"]["enterprise"] == 0

    def test_get_stats_with_activity(self, fresh_tier_limiter):
        """Should return accurate stats with activity."""
        fresh_tier_limiter.allow("user-1", tier="free")
        fresh_tier_limiter.allow("user-2", tier="free")
        fresh_tier_limiter.allow("user-1", tier="professional")

        stats = fresh_tier_limiter.get_stats()

        assert stats["tier_buckets"]["free"] == 2
        assert stats["tier_buckets"]["professional"] == 1


# -----------------------------------------------------------------------------
# Reset Tests
# -----------------------------------------------------------------------------


class TestReset:
    """Tests for reset functionality."""

    def test_reset_clears_all_tier_buckets(self, fresh_tier_limiter):
        """reset() should clear all tier buckets."""
        fresh_tier_limiter.allow("user-1", tier="free")
        fresh_tier_limiter.allow("user-2", tier="enterprise")

        fresh_tier_limiter.reset()

        stats = fresh_tier_limiter.get_stats()
        for tier, count in stats["tier_buckets"].items():
            assert count == 0


# -----------------------------------------------------------------------------
# Global Limiter Tests
# -----------------------------------------------------------------------------


class TestGlobalLimiter:
    """Tests for global tier rate limiter instance."""

    def test_get_tier_rate_limiter_singleton(self, clean_tier_limiter):
        """get_tier_rate_limiter should return singleton."""
        from aragora.server.middleware.rate_limit.tier_limiter import (
            get_tier_rate_limiter,
        )

        limiter1 = get_tier_rate_limiter()
        limiter2 = get_tier_rate_limiter()

        assert limiter1 is limiter2


# -----------------------------------------------------------------------------
# check_tier_rate_limit Helper Tests
# -----------------------------------------------------------------------------


class TestCheckTierRateLimit:
    """Tests for check_tier_rate_limit helper function."""

    def test_check_with_anonymous_user(self, clean_tier_limiter):
        """Should use free tier for anonymous users."""
        from aragora.server.middleware.rate_limit.tier_limiter import (
            check_tier_rate_limit,
        )

        handler = MagicMock()
        handler.client_address = ("192.168.1.100", 12345)
        handler.headers = MagicMock()
        handler.headers.get = MagicMock(return_value="")

        result = check_tier_rate_limit(handler)

        assert result.allowed is True
        # Should use free tier (10 rpm)
        assert result.limit == 10

    def test_check_with_authenticated_user_and_org(self, clean_tier_limiter):
        """Should use org tier for authenticated users."""
        from aragora.server.middleware.rate_limit.tier_limiter import (
            check_tier_rate_limit,
        )

        handler = MagicMock()
        handler.client_address = ("192.168.1.100", 12345)
        handler.headers = MagicMock()
        handler.headers.get = MagicMock(return_value="")

        user_store = MagicMock()

        # Mock org with professional tier
        mock_org = MagicMock()
        mock_org.tier.value = "professional"
        user_store.get_organization_by_id.return_value = mock_org

        # Patch at the actual import location in billing.jwt_auth
        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_auth = MagicMock()
            mock_auth.is_authenticated = True
            mock_auth.user_id = "user-123"
            mock_auth.org_id = "org-456"
            mock_extract.return_value = mock_auth

            result = check_tier_rate_limit(handler, user_store)

            assert result.allowed is True
            assert result.limit == 200  # Professional tier

    def test_check_fallback_when_extraction_fails(self, clean_tier_limiter):
        """Should fallback to free tier when auth extraction fails."""
        from aragora.server.middleware.rate_limit.tier_limiter import (
            check_tier_rate_limit,
        )

        handler = MagicMock()
        handler.client_address = ("192.168.1.100", 12345)
        handler.headers = MagicMock()
        handler.headers.get = MagicMock(return_value="")

        user_store = MagicMock()

        # Patch at the actual import location in billing.jwt_auth
        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.side_effect = RuntimeError("Auth failed")

            result = check_tier_rate_limit(handler, user_store)

            assert result.allowed is True
            assert result.limit == 10  # Free tier

    def test_check_uses_user_id_as_key_when_authenticated(self, clean_tier_limiter):
        """Should use user_id as key for authenticated users."""
        from aragora.server.middleware.rate_limit.tier_limiter import (
            check_tier_rate_limit,
        )

        handler = MagicMock()
        handler.client_address = ("192.168.1.100", 12345)
        handler.headers = MagicMock()
        handler.headers.get = MagicMock(return_value="")

        user_store = MagicMock()
        user_store.get_organization_by_id.return_value = None  # No org

        # Patch at the actual import location in billing.jwt_auth
        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_auth = MagicMock()
            mock_auth.is_authenticated = True
            mock_auth.user_id = "authenticated-user"
            mock_auth.org_id = None
            mock_extract.return_value = mock_auth

            result = check_tier_rate_limit(handler, user_store)

            assert "authenticated-user" in result.key


# -----------------------------------------------------------------------------
# Retry After Tests
# -----------------------------------------------------------------------------


class TestRetryAfter:
    """Tests for retry_after behavior."""

    def test_returns_retry_after_when_denied(self):
        """Should return retry_after when request denied."""
        from aragora.server.middleware.rate_limit.tier_limiter import TierRateLimiter

        limiter = TierRateLimiter(tier_limits={"limited": (1, 2)})

        # Exhaust limit
        for _ in range(10):
            limiter.allow("user-123", tier="limited")

        result = limiter.allow("user-123", tier="limited")

        if not result.allowed:
            assert result.retry_after > 0

    def test_no_retry_after_when_allowed(self, fresh_tier_limiter):
        """Should not have retry_after when request allowed."""
        result = fresh_tier_limiter.allow("user-123", tier="enterprise")

        assert result.allowed is True
        assert result.retry_after == 0


# -----------------------------------------------------------------------------
# Thread Safety Tests
# -----------------------------------------------------------------------------


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_requests_thread_safe(self, fresh_tier_limiter):
        """Concurrent requests should be thread-safe."""
        results = []
        errors = []

        def make_requests():
            try:
                for _ in range(50):
                    result = fresh_tier_limiter.allow("user-shared", tier="enterprise")
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

    def test_concurrent_different_tiers(self, fresh_tier_limiter):
        """Concurrent requests across different tiers should work."""
        results_per_tier = {}
        lock = threading.Lock()

        def make_requests(tier):
            results = []
            for _ in range(10):
                result = fresh_tier_limiter.allow(f"user-{tier}", tier=tier)
                results.append(result)
            with lock:
                results_per_tier[tier] = results

        tiers = ["free", "starter", "professional", "enterprise"]
        threads = [threading.Thread(target=make_requests, args=(t,)) for t in tiers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All tiers should have results
        assert len(results_per_tier) == 4


# -----------------------------------------------------------------------------
# Edge Cases Tests
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_client_key(self, fresh_tier_limiter):
        """Should handle empty client_key."""
        result = fresh_tier_limiter.allow("", tier="free")
        assert result.allowed is True

    def test_empty_tier_defaults_to_free(self, fresh_tier_limiter):
        """Empty tier should use free tier limits."""
        result = fresh_tier_limiter.allow("user-123", tier="")

        # Empty string lowercased is still empty, defaults to free
        assert result.limit == 10  # Free tier rate

    def test_very_long_client_key(self, fresh_tier_limiter):
        """Should handle very long client_key."""
        long_key = "user-" + "a" * 10000
        result = fresh_tier_limiter.allow(long_key, tier="free")
        assert result.allowed is True

    def test_special_characters_in_client_key(self, fresh_tier_limiter):
        """Should handle special characters in client_key."""
        result = fresh_tier_limiter.allow("user:with:colons@special.chars", tier="free")
        assert result.allowed is True

    def test_handler_without_client_address(self, clean_tier_limiter):
        """Should handle handler without client_address."""
        from aragora.server.middleware.rate_limit.tier_limiter import (
            check_tier_rate_limit,
        )

        handler = MagicMock(spec=[])  # No attributes

        result = check_tier_rate_limit(handler)

        assert result.allowed is True

    def test_handler_with_tuple_client_address_empty(self, clean_tier_limiter):
        """Should handle empty tuple client_address."""
        from aragora.server.middleware.rate_limit.tier_limiter import (
            check_tier_rate_limit,
        )

        handler = MagicMock()
        handler.client_address = ()
        handler.headers = MagicMock()
        handler.headers.get = MagicMock(return_value="")

        result = check_tier_rate_limit(handler)

        assert result.allowed is True

    def test_new_tier_added_dynamically(self):
        """Should handle dynamically added tiers."""
        from aragora.server.middleware.rate_limit.tier_limiter import TierRateLimiter

        limiter = TierRateLimiter(
            tier_limits={
                "free": (10, 20),
                "vip": (500, 1000),  # New custom tier
            }
        )

        result = limiter.allow("user-123", tier="vip")

        assert result.allowed is True
        assert result.limit == 500


# -----------------------------------------------------------------------------
# Configuration Validation Tests
# -----------------------------------------------------------------------------


class TestConfigurationValidation:
    """Tests for configuration validation."""

    def test_tier_limits_are_tuples(self):
        """Tier limits should be (rate, burst) tuples."""
        from aragora.server.middleware.rate_limit.tier_limiter import TIER_RATE_LIMITS

        for tier, limits in TIER_RATE_LIMITS.items():
            assert isinstance(limits, tuple), f"{tier} limits not a tuple"
            assert len(limits) == 2, f"{tier} limits should have 2 elements"
            assert limits[0] > 0, f"{tier} rate should be positive"
            assert limits[1] > 0, f"{tier} burst should be positive"

    def test_tiers_ordered_by_rate(self):
        """Tiers should be ordered from lowest to highest rate."""
        from aragora.server.middleware.rate_limit.tier_limiter import TIER_RATE_LIMITS

        rates = [
            TIER_RATE_LIMITS["free"][0],
            TIER_RATE_LIMITS["starter"][0],
            TIER_RATE_LIMITS["professional"][0],
            TIER_RATE_LIMITS["enterprise"][0],
        ]

        assert rates == sorted(rates), "Tiers should be in ascending rate order"
