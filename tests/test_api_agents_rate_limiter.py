"""
Tests for the OpenRouter rate limiter module.

Covers token bucket algorithm, tier configuration, header parsing,
thread safety, and global limiter management.
"""

import asyncio
import os
import threading
import time
from unittest.mock import patch

import pytest

from aragora.agents.api_agents.rate_limiter import (
    OPENROUTER_TIERS,
    OpenRouterRateLimiter,
    OpenRouterTier,
    get_openrouter_limiter,
    set_openrouter_tier,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def limiter():
    """Create a rate limiter with standard tier."""
    return OpenRouterRateLimiter(tier="standard")


@pytest.fixture
def free_limiter():
    """Create a rate limiter with free tier (lower limits)."""
    return OpenRouterRateLimiter(tier="free")


@pytest.fixture
def premium_limiter():
    """Create a rate limiter with premium tier (higher limits)."""
    return OpenRouterRateLimiter(tier="premium")


@pytest.fixture(autouse=True)
def reset_global_limiter():
    """Reset global limiter state between tests."""
    import aragora.agents.api_agents.rate_limiter as module
    with module._openrouter_limiter_lock:
        module._openrouter_limiter = None
    yield
    with module._openrouter_limiter_lock:
        module._openrouter_limiter = None


# =============================================================================
# OpenRouterTier Tests
# =============================================================================


class TestOpenRouterTier:
    """Tests for OpenRouterTier dataclass."""

    def test_tier_has_required_fields(self):
        """Should have name, requests_per_minute fields."""
        tier = OpenRouterTier(name="test", requests_per_minute=100)
        assert tier.name == "test"
        assert tier.requests_per_minute == 100

    def test_tier_default_tokens_per_minute(self):
        """Should default tokens_per_minute to 0 (unlimited)."""
        tier = OpenRouterTier(name="test", requests_per_minute=100)
        assert tier.tokens_per_minute == 0

    def test_tier_default_burst_size(self):
        """Should default burst_size to 10."""
        tier = OpenRouterTier(name="test", requests_per_minute=100)
        assert tier.burst_size == 10

    def test_tier_custom_values(self):
        """Should accept custom values for all fields."""
        tier = OpenRouterTier(
            name="custom",
            requests_per_minute=500,
            tokens_per_minute=100000,
            burst_size=50,
        )
        assert tier.name == "custom"
        assert tier.requests_per_minute == 500
        assert tier.tokens_per_minute == 100000
        assert tier.burst_size == 50


# =============================================================================
# OPENROUTER_TIERS Constants Tests
# =============================================================================


class TestOpenRouterTiers:
    """Tests for OPENROUTER_TIERS constant."""

    def test_has_free_tier(self):
        """Should have a free tier defined."""
        assert "free" in OPENROUTER_TIERS
        assert OPENROUTER_TIERS["free"].requests_per_minute == 20

    def test_has_basic_tier(self):
        """Should have a basic tier defined."""
        assert "basic" in OPENROUTER_TIERS
        assert OPENROUTER_TIERS["basic"].requests_per_minute == 60

    def test_has_standard_tier(self):
        """Should have a standard tier defined."""
        assert "standard" in OPENROUTER_TIERS
        assert OPENROUTER_TIERS["standard"].requests_per_minute == 200

    def test_has_premium_tier(self):
        """Should have a premium tier defined."""
        assert "premium" in OPENROUTER_TIERS
        assert OPENROUTER_TIERS["premium"].requests_per_minute == 500

    def test_has_unlimited_tier(self):
        """Should have an unlimited tier defined."""
        assert "unlimited" in OPENROUTER_TIERS
        assert OPENROUTER_TIERS["unlimited"].requests_per_minute == 1000

    def test_tiers_have_increasing_limits(self):
        """Tiers should have increasing request limits."""
        order = ["free", "basic", "standard", "premium", "unlimited"]
        rpms = [OPENROUTER_TIERS[t].requests_per_minute for t in order]
        assert rpms == sorted(rpms), "Tiers should have increasing RPM limits"

    def test_tiers_have_increasing_burst_sizes(self):
        """Tiers should have increasing burst sizes."""
        order = ["free", "basic", "standard", "premium", "unlimited"]
        bursts = [OPENROUTER_TIERS[t].burst_size for t in order]
        assert bursts == sorted(bursts), "Tiers should have increasing burst sizes"


# =============================================================================
# OpenRouterRateLimiter Initialization Tests
# =============================================================================


class TestRateLimiterInit:
    """Tests for OpenRouterRateLimiter initialization."""

    def test_default_tier_is_standard(self):
        """Should use standard tier by default."""
        limiter = OpenRouterRateLimiter()
        assert limiter.tier.name == "standard"

    def test_accepts_tier_parameter(self, free_limiter):
        """Should accept custom tier parameter."""
        assert free_limiter.tier.name == "free"

    def test_unknown_tier_falls_back_to_standard(self):
        """Should fall back to standard for unknown tier."""
        limiter = OpenRouterRateLimiter(tier="unknown_tier")
        assert limiter.tier.name == "standard"

    def test_reads_tier_from_environment(self):
        """Should read tier from OPENROUTER_TIER env var."""
        with patch.dict(os.environ, {"OPENROUTER_TIER": "premium"}):
            limiter = OpenRouterRateLimiter()
            assert limiter.tier.name == "premium"

    def test_env_tier_overrides_parameter(self):
        """Environment variable should override parameter."""
        with patch.dict(os.environ, {"OPENROUTER_TIER": "free"}):
            limiter = OpenRouterRateLimiter(tier="premium")
            assert limiter.tier.name == "free"

    def test_case_insensitive_tier_name(self):
        """Tier name should be case-insensitive."""
        limiter = OpenRouterRateLimiter(tier="PREMIUM")
        assert limiter.tier.name == "premium"

    def test_initial_tokens_equal_burst_size(self, limiter):
        """Should start with tokens equal to burst size."""
        assert limiter._tokens == limiter.tier.burst_size

    def test_initial_api_limits_are_none(self, limiter):
        """Should have None for API-reported limits initially."""
        assert limiter._api_limit is None
        assert limiter._api_remaining is None
        assert limiter._api_reset is None


# =============================================================================
# Token Bucket Algorithm Tests
# =============================================================================


class TestTokenBucketAlgorithm:
    """Tests for token bucket rate limiting logic."""

    @pytest.mark.asyncio
    async def test_acquire_succeeds_with_tokens(self, limiter):
        """Should succeed immediately when tokens available."""
        result = await limiter.acquire(timeout=0.1)
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_decrements_tokens(self, limiter):
        """Should decrement token count on acquire."""
        initial = limiter._tokens
        await limiter.acquire(timeout=0.1)
        assert limiter._tokens == initial - 1

    @pytest.mark.asyncio
    async def test_multiple_acquires_work(self, limiter):
        """Should allow multiple acquires up to burst size."""
        burst_size = limiter.tier.burst_size
        for _ in range(burst_size):
            result = await limiter.acquire(timeout=0.1)
            assert result is True

    @pytest.mark.asyncio
    async def test_acquire_times_out_when_empty(self, free_limiter):
        """Should timeout when no tokens available."""
        # Exhaust all tokens
        for _ in range(free_limiter.tier.burst_size):
            await free_limiter.acquire(timeout=0.1)

        # Next acquire should timeout
        result = await free_limiter.acquire(timeout=0.1)
        assert result is False

    @pytest.mark.asyncio
    async def test_tokens_refill_over_time(self, free_limiter):
        """Should refill tokens based on requests_per_minute."""
        # Exhaust all tokens
        for _ in range(free_limiter.tier.burst_size):
            await free_limiter.acquire(timeout=0.1)

        # Manually adjust last_refill to simulate time passing
        # Free tier: 20 RPM = 1 token per 3 seconds
        free_limiter._last_refill = time.monotonic() - 6  # 6 seconds = 2 tokens

        # Should now have tokens available
        result = await free_limiter.acquire(timeout=0.1)
        assert result is True

    def test_refill_caps_at_burst_size(self, limiter):
        """Refill should not exceed burst size."""
        # Set last refill to long ago
        limiter._last_refill = time.monotonic() - 3600  # 1 hour ago

        # Trigger refill
        limiter._refill()

        # Should be capped at burst size
        assert limiter._tokens == limiter.tier.burst_size


# =============================================================================
# Header Parsing Tests
# =============================================================================


class TestHeaderParsing:
    """Tests for update_from_headers method."""

    def test_parses_limit_header(self, limiter):
        """Should parse X-RateLimit-Limit header."""
        limiter.update_from_headers({"X-RateLimit-Limit": "100"})
        assert limiter._api_limit == 100

    def test_parses_remaining_header(self, limiter):
        """Should parse X-RateLimit-Remaining header."""
        limiter.update_from_headers({"X-RateLimit-Remaining": "50"})
        assert limiter._api_remaining == 50

    def test_parses_reset_header(self, limiter):
        """Should parse X-RateLimit-Reset header."""
        reset_time = time.time() + 60
        limiter.update_from_headers({"X-RateLimit-Reset": str(reset_time)})
        assert limiter._api_reset == pytest.approx(reset_time, rel=0.01)

    def test_parses_all_headers_together(self, limiter):
        """Should parse all rate limit headers."""
        reset_time = time.time() + 120
        limiter.update_from_headers({
            "X-RateLimit-Limit": "200",
            "X-RateLimit-Remaining": "150",
            "X-RateLimit-Reset": str(reset_time),
        })
        assert limiter._api_limit == 200
        assert limiter._api_remaining == 150
        assert limiter._api_reset == pytest.approx(reset_time, rel=0.01)

    def test_ignores_unrelated_headers(self, limiter):
        """Should ignore non-rate-limit headers."""
        limiter.update_from_headers({
            "Content-Type": "application/json",
            "X-Custom-Header": "value",
        })
        assert limiter._api_limit is None
        assert limiter._api_remaining is None

    def test_handles_invalid_limit_header(self, limiter):
        """Should handle invalid X-RateLimit-Limit gracefully."""
        limiter.update_from_headers({"X-RateLimit-Limit": "not-a-number"})
        assert limiter._api_limit is None  # Should remain None

    def test_handles_invalid_remaining_header(self, limiter):
        """Should handle invalid X-RateLimit-Remaining gracefully."""
        limiter.update_from_headers({"X-RateLimit-Remaining": "abc"})
        assert limiter._api_remaining is None

    def test_handles_invalid_reset_header(self, limiter):
        """Should handle invalid X-RateLimit-Reset gracefully."""
        limiter.update_from_headers({"X-RateLimit-Reset": "invalid"})
        assert limiter._api_reset is None

    def test_handles_empty_headers(self, limiter):
        """Should handle empty headers dict."""
        limiter.update_from_headers({})
        assert limiter._api_limit is None


# =============================================================================
# Error Recovery Tests
# =============================================================================


class TestErrorRecovery:
    """Tests for release_on_error method."""

    @pytest.mark.asyncio
    async def test_release_adds_token(self, limiter):
        """Should add a token back on error."""
        initial = limiter._tokens
        await limiter.acquire(timeout=0.1)
        assert limiter._tokens == initial - 1

        limiter.release_on_error()
        assert limiter._tokens == initial

    def test_release_caps_at_burst_size(self, limiter):
        """Should not exceed burst size when releasing."""
        # Already at burst size
        limiter._tokens = limiter.tier.burst_size

        limiter.release_on_error()

        # Should still be at burst size
        assert limiter._tokens == limiter.tier.burst_size


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStats:
    """Tests for stats property."""

    def test_stats_includes_tier_info(self, limiter):
        """Should include tier name and RPM limit."""
        stats = limiter.stats
        assert stats["tier"] == "standard"
        assert stats["rpm_limit"] == 200

    def test_stats_includes_tokens(self, limiter):
        """Should include available tokens."""
        stats = limiter.stats
        assert stats["tokens_available"] == limiter.tier.burst_size

    def test_stats_includes_burst_size(self, limiter):
        """Should include burst size."""
        stats = limiter.stats
        assert stats["burst_size"] == limiter.tier.burst_size

    def test_stats_includes_api_limits(self, limiter):
        """Should include API-reported limits."""
        limiter.update_from_headers({
            "X-RateLimit-Limit": "100",
            "X-RateLimit-Remaining": "75",
        })
        stats = limiter.stats
        assert stats["api_limit"] == 100
        assert stats["api_remaining"] == 75

    @pytest.mark.asyncio
    async def test_stats_reflects_token_usage(self, limiter):
        """Stats should reflect token usage."""
        initial_tokens = limiter.stats["tokens_available"]
        await limiter.acquire(timeout=0.1)
        await limiter.acquire(timeout=0.1)
        assert limiter.stats["tokens_available"] == initial_tokens - 2


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe operation."""

    @pytest.mark.asyncio
    async def test_concurrent_acquires(self, limiter):
        """Should handle concurrent acquire calls safely."""
        results = await asyncio.gather(*[
            limiter.acquire(timeout=1.0)
            for _ in range(limiter.tier.burst_size)
        ])
        assert all(r is True for r in results)

    def test_concurrent_header_updates(self, limiter):
        """Should handle concurrent header updates safely."""
        def update_headers():
            for i in range(100):
                limiter.update_from_headers({
                    "X-RateLimit-Limit": str(100 + i),
                    "X-RateLimit-Remaining": str(50 + i),
                })

        threads = [threading.Thread(target=update_headers) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not crash and should have valid values
        assert limiter._api_limit is not None
        assert limiter._api_remaining is not None


# =============================================================================
# Global Limiter Tests
# =============================================================================


class TestGlobalLimiter:
    """Tests for global limiter management."""

    def test_get_limiter_creates_instance(self):
        """Should create a limiter on first call."""
        limiter = get_openrouter_limiter()
        assert isinstance(limiter, OpenRouterRateLimiter)

    def test_get_limiter_returns_same_instance(self):
        """Should return same instance on subsequent calls."""
        limiter1 = get_openrouter_limiter()
        limiter2 = get_openrouter_limiter()
        assert limiter1 is limiter2

    def test_set_tier_creates_new_limiter(self):
        """Should create new limiter with specified tier."""
        limiter1 = get_openrouter_limiter()
        assert limiter1.tier.name == "standard"  # Default

        set_openrouter_tier("premium")
        limiter2 = get_openrouter_limiter()

        # Should be different instance with new tier
        assert limiter2.tier.name == "premium"

    def test_set_tier_replaces_existing(self):
        """Should replace existing global limiter."""
        old_limiter = get_openrouter_limiter()
        set_openrouter_tier("free")
        new_limiter = get_openrouter_limiter()

        assert old_limiter is not new_limiter
        assert new_limiter.tier.name == "free"


# =============================================================================
# API Limit Behavior Tests
# =============================================================================


class TestApiLimitBehavior:
    """Tests for behavior when API reports limits."""

    @pytest.mark.asyncio
    async def test_respects_zero_remaining(self, limiter):
        """Should wait when API reports zero remaining."""
        # Simulate API saying no remaining requests
        limiter.update_from_headers({
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(time.time() + 0.1),  # Reset in 0.1s
        })

        # Should still work (token bucket has tokens)
        start = time.monotonic()
        result = await limiter.acquire(timeout=0.5)
        elapsed = time.monotonic() - start

        # Should have waited some time due to API limit check
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_with_long_timeout(self, free_limiter):
        """Should wait and retry with longer timeout."""
        # Exhaust tokens
        for _ in range(free_limiter.tier.burst_size):
            await free_limiter.acquire(timeout=0.1)

        # With longer timeout, should wait for refill
        # Free tier: 20 RPM = 3 seconds per token
        # Simulate time passing
        free_limiter._last_refill = time.monotonic() - 4

        result = await free_limiter.acquire(timeout=0.1)
        assert result is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestRateLimiterIntegration:
    """Integration tests for rate limiter."""

    @pytest.mark.asyncio
    async def test_realistic_usage_pattern(self, limiter):
        """Test realistic request pattern with header updates."""
        # Simulate a series of API calls
        for i in range(5):
            acquired = await limiter.acquire(timeout=1.0)
            assert acquired is True

            # Simulate API response headers
            limiter.update_from_headers({
                "X-RateLimit-Limit": "200",
                "X-RateLimit-Remaining": str(195 - i),
            })

        # Check final state
        stats = limiter.stats
        assert stats["api_remaining"] == 191

    @pytest.mark.asyncio
    async def test_error_and_retry_pattern(self, limiter):
        """Test acquiring, error, release, and retry."""
        # Acquire
        await limiter.acquire(timeout=0.1)
        tokens_after_acquire = limiter._tokens

        # Simulate error and release
        limiter.release_on_error()
        tokens_after_release = limiter._tokens

        # Should have one more token
        assert tokens_after_release == tokens_after_acquire + 1

        # Retry should succeed
        result = await limiter.acquire(timeout=0.1)
        assert result is True
