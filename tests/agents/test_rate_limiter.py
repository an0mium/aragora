"""
Comprehensive tests for aragora.agents.api_agents.rate_limiter module.

Additional test coverage beyond api_agents/test_rate_limiter.py including:
- Token bucket algorithm verification
- Sliding window behavior
- Multi-provider concurrent handling
- Stress testing and race conditions
- Backoff edge cases
- Header parsing edge cases
- Memory/resource management
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from aragora.agents.api_agents.rate_limiter import (
    OPENROUTER_TIERS,
    PROVIDER_DEFAULT_TIERS,
    OpenRouterRateLimiter,
    OpenRouterTier,
    ProviderRateLimiter,
    ProviderRateLimiterRegistry,
    ProviderRateLimitContext,
    ProviderTier,
    RateLimitContext,
    get_openrouter_limiter,
    get_provider_limiter,
    get_provider_registry,
    reset_provider_limiters,
    set_openrouter_tier,
)
from aragora.shared.rate_limiting import ExponentialBackoff, TokenBucket


# ============================================================================
# Token Bucket Algorithm Tests
# ============================================================================


class TestTokenBucketAlgorithm:
    """Tests verifying correct token bucket algorithm implementation."""

    def test_token_refill_rate_calculation(self):
        """Verify tokens refill at correct rate per minute."""
        limiter = ProviderRateLimiter("test", rpm=120, burst=10)  # 2 per second
        limiter._tokens = 0
        limiter._last_refill = time.monotonic() - 1.0  # 1 second ago

        limiter._refill()

        # Should have added 2 tokens (120 per minute = 2 per second)
        assert 1.9 <= limiter._tokens <= 2.1

    def test_burst_capacity_limits_accumulation(self):
        """Verify tokens don't accumulate beyond burst capacity."""
        limiter = ProviderRateLimiter("test", rpm=1000, burst=5)
        limiter._tokens = 5
        limiter._last_refill = time.monotonic() - 120  # 2 minutes ago

        limiter._refill()

        # Tokens should still be capped at burst size
        assert limiter._tokens == 5

    def test_fractional_token_handling(self):
        """Verify fractional tokens are handled correctly."""
        limiter = ProviderRateLimiter("test", rpm=60, burst=10)  # 1 per second
        limiter._tokens = 0
        limiter._last_refill = time.monotonic() - 0.5  # 0.5 seconds ago

        limiter._refill()

        # Should have ~0.5 tokens
        assert 0.4 <= limiter._tokens <= 0.6

    def test_token_consumption_is_atomic(self):
        """Verify token consumption happens atomically."""
        limiter = ProviderRateLimiter("test", rpm=60, burst=3)
        limiter._tokens = 1.5
        limiter._last_refill = time.monotonic()  # Freeze time

        # Should consume exactly 1 token
        initial = limiter._tokens
        limiter._bucket._sync_lock.acquire()
        try:
            limiter._bucket._refill()
            if limiter._bucket._tokens >= 1.0:
                limiter._bucket._tokens -= 1.0
                consumed = True
            else:
                consumed = False
        finally:
            limiter._bucket._sync_lock.release()

        assert consumed is True
        assert limiter._tokens == initial - 1.0


# ============================================================================
# Rate Limit Enforcement Tests
# ============================================================================


class TestRateLimitEnforcement:
    """Tests verifying rate limits are properly enforced."""

    @pytest.mark.asyncio
    async def test_burst_allows_rapid_requests(self):
        """Burst capacity allows rapid initial requests."""
        limiter = ProviderRateLimiter("test", rpm=60, burst=5)

        # Should be able to make 5 rapid requests
        results = []
        for _ in range(5):
            results.append(await limiter.acquire(timeout=0.1))

        assert all(results)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_exhausted_burst_causes_wait(self):
        """After burst is exhausted, must wait for token refill."""
        limiter = ProviderRateLimiter("test", rpm=60, burst=2)

        # Exhaust burst
        await limiter.acquire(timeout=0.1)
        await limiter.acquire(timeout=0.1)

        # Next request should take time or fail quickly
        start = time.monotonic()
        result = await limiter.acquire(timeout=0.1)
        elapsed = time.monotonic() - start

        # Either waited or timed out
        assert elapsed >= 0.05 or result is False

    @pytest.mark.asyncio
    async def test_rate_recovery_over_time(self):
        """Verify rate limit recovers as time passes."""
        limiter = ProviderRateLimiter("test", rpm=600, burst=1)  # 10 per second

        # Consume the token
        await limiter.acquire(timeout=0.1)

        # Wait for recovery (0.1s = 1 token)
        await asyncio.sleep(0.15)

        # Should be able to acquire again
        result = await limiter.acquire(timeout=0.1)
        assert result is True

    @pytest.mark.asyncio
    async def test_sustained_rate_below_limit(self):
        """Sustained requests below rate limit should all succeed."""
        limiter = ProviderRateLimiter("test", rpm=600, burst=10)  # 10 per second

        results = []
        for _ in range(5):
            results.append(await limiter.acquire(timeout=1.0))
            await asyncio.sleep(0.15)  # Below rate limit

        assert all(results)


# ============================================================================
# Multi-Provider Handling Tests
# ============================================================================


class TestMultiProviderHandling:
    """Tests for handling multiple API providers simultaneously."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Reset provider limiters before and after each test."""
        reset_provider_limiters()
        yield
        reset_provider_limiters()

    def test_separate_limiters_per_provider(self):
        """Each provider gets independent rate limits."""
        anthropic = get_provider_limiter("anthropic")
        openai = get_provider_limiter("openai")

        assert anthropic is not openai
        assert anthropic.provider == "anthropic"
        assert openai.provider == "openai"

    def test_provider_limits_independent(self):
        """Rate limiting one provider doesn't affect others."""
        anthropic = get_provider_limiter("anthropic")
        openai = get_provider_limiter("openai")

        # Exhaust anthropic's tokens
        anthropic._tokens = 0

        # OpenAI should still have tokens
        assert openai._tokens > 0

    @pytest.mark.asyncio
    async def test_concurrent_multi_provider_requests(self):
        """Can make concurrent requests to different providers."""
        reset_provider_limiters()
        providers = ["anthropic", "openai", "gemini", "mistral"]
        limiters = [get_provider_limiter(p) for p in providers]

        async def acquire_from_provider(limiter):
            return await limiter.acquire(timeout=1.0)

        results = await asyncio.gather(*[acquire_from_provider(lim) for lim in limiters])

        assert all(results)
        assert len(results) == 4

    def test_provider_specific_defaults(self):
        """Each provider uses its own default limits."""
        anthropic = get_provider_limiter("anthropic")
        gemini = get_provider_limiter("gemini")

        assert (
            anthropic.requests_per_minute == PROVIDER_DEFAULT_TIERS["anthropic"].requests_per_minute
        )
        assert gemini.requests_per_minute == PROVIDER_DEFAULT_TIERS["gemini"].requests_per_minute
        assert anthropic.requests_per_minute != gemini.requests_per_minute


# ============================================================================
# Exponential Backoff Tests
# ============================================================================


class TestExponentialBackoff:
    """Tests for exponential backoff behavior."""

    def test_initial_backoff_is_base_delay(self):
        """First backoff uses base delay."""
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=60.0, jitter=0.0)

        delay = backoff.record_failure()

        # First failure: 1.0 * 2^1 = 2.0
        assert delay == 2.0

    def test_backoff_doubles_exponentially(self):
        """Each failure roughly doubles the delay."""
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=1000.0, jitter=0.0)

        delays = []
        for _ in range(5):
            delays.append(backoff.record_failure())

        # Should be 2, 4, 8, 16, 32 (1.0 * 2^i for i=1..5)
        expected = [2.0, 4.0, 8.0, 16.0, 32.0]
        assert delays == expected

    def test_backoff_caps_at_max_delay(self):
        """Backoff delay caps at max_delay."""
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=10.0, jitter=0.0)

        delays = []
        for _ in range(10):
            delays.append(backoff.record_failure())

        # All delays after the cap should be max_delay
        assert all(d <= 10.0 for d in delays)
        assert delays[-1] == 10.0

    def test_jitter_adds_randomness(self):
        """Jitter adds random variance to delays."""
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=60.0, jitter=0.5)

        # Collect multiple delays
        delays = set()
        for _ in range(10):
            backoff.failure_count = 0
            delays.add(backoff.record_failure())

        # With 50% jitter, delays should vary
        # Base delay is 2.0, jitter adds 0 to 1.0
        assert len(delays) > 1  # Should have some variation

    def test_reset_clears_failure_count(self):
        """Reset clears failure count."""
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=60.0)

        for _ in range(5):
            backoff.record_failure()

        assert backoff.failure_count == 5

        backoff.reset()

        assert backoff.failure_count == 0
        assert backoff.is_backing_off is False

    def test_is_backing_off_property(self):
        """is_backing_off reflects failure state."""
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=60.0)

        assert backoff.is_backing_off is False

        backoff.record_failure()
        assert backoff.is_backing_off is True

        backoff.reset()
        assert backoff.is_backing_off is False


# ============================================================================
# OpenRouter Tier Tests
# ============================================================================


class TestOpenRouterTiers:
    """Tests for OpenRouter tier configurations."""

    def test_free_tier_has_lowest_limits(self):
        """Free tier has lowest limits."""
        free = OPENROUTER_TIERS["free"]
        basic = OPENROUTER_TIERS["basic"]

        assert free.requests_per_minute < basic.requests_per_minute
        assert free.burst_size < basic.burst_size

    def test_unlimited_tier_has_highest_limits(self):
        """Unlimited tier has highest limits."""
        unlimited = OPENROUTER_TIERS["unlimited"]

        for name, tier in OPENROUTER_TIERS.items():
            if name != "unlimited":
                assert unlimited.requests_per_minute >= tier.requests_per_minute

    def test_tier_selection_case_insensitive(self):
        """Tier selection is case-insensitive."""
        limiter1 = OpenRouterRateLimiter(tier="PREMIUM")
        limiter2 = OpenRouterRateLimiter(tier="premium")
        limiter3 = OpenRouterRateLimiter(tier="Premium")

        assert limiter1.tier.name == "premium"
        assert limiter2.tier.name == "premium"
        assert limiter3.tier.name == "premium"

    def test_invalid_tier_defaults_to_standard(self):
        """Invalid tier name defaults to standard."""
        limiter = OpenRouterRateLimiter(tier="nonexistent_tier")

        assert limiter.tier.name == "standard"


# ============================================================================
# Header Parsing Tests
# ============================================================================


class TestHeaderParsing:
    """Tests for rate limit header parsing."""

    def test_standard_header_format(self):
        """Parse standard X-RateLimit headers."""
        limiter = ProviderRateLimiter("test")

        headers = {
            "X-RateLimit-Limit": "1000",
            "X-RateLimit-Remaining": "500",
            "X-RateLimit-Reset": str(time.time() + 60),
        }

        limiter.update_from_headers(headers)

        assert limiter._api_limit == 1000
        assert limiter._api_remaining == 500
        assert limiter._api_reset is not None

    def test_lowercase_header_format(self):
        """Parse lowercase header format."""
        limiter = ProviderRateLimiter("test")

        headers = {
            "x-ratelimit-limit": "2000",
            "x-ratelimit-remaining": "1500",
        }

        limiter.update_from_headers(headers)

        assert limiter._api_limit == 2000
        assert limiter._api_remaining == 1500

    def test_ratelimit_header_format(self):
        """Parse RateLimit- prefix format."""
        limiter = ProviderRateLimiter("test")

        headers = {
            "RateLimit-Limit": "3000",
            "RateLimit-Remaining": "2500",
        }

        limiter.update_from_headers(headers)

        assert limiter._api_limit == 3000
        assert limiter._api_remaining == 2500

    def test_mixed_case_headers_not_matched(self):
        """Only specific case formats are matched."""
        limiter = ProviderRateLimiter("test")

        # This non-standard format should not match
        headers = {
            "X-RATELIMIT-LIMIT": "4000",
        }

        limiter.update_from_headers(headers)

        # Should remain None
        assert limiter._api_limit is None

    def test_non_numeric_values_ignored(self):
        """Non-numeric header values are ignored."""
        limiter = ProviderRateLimiter("test")

        headers = {
            "X-RateLimit-Limit": "invalid",
            "X-RateLimit-Remaining": "abc123",
        }

        limiter.update_from_headers(headers)

        assert limiter._api_limit is None
        assert limiter._api_remaining is None

    def test_partial_headers_update(self):
        """Partial header updates work correctly."""
        limiter = ProviderRateLimiter("test")

        # First update with limit only
        limiter.update_from_headers({"X-RateLimit-Limit": "100"})
        assert limiter._api_limit == 100
        assert limiter._api_remaining is None

        # Second update with remaining only
        limiter.update_from_headers({"X-RateLimit-Remaining": "50"})
        assert limiter._api_limit == 100  # Preserved
        assert limiter._api_remaining == 50


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestContextManagers:
    """Tests for rate limit context managers."""

    @pytest.mark.asyncio
    async def test_rate_limit_context_success(self):
        """RateLimitContext handles success case."""
        limiter = OpenRouterRateLimiter(tier="unlimited")

        async with limiter.request(timeout=1.0) as ctx:
            assert bool(ctx) is True
            assert ctx._acquired is True

    @pytest.mark.asyncio
    async def test_rate_limit_context_timeout(self):
        """RateLimitContext handles timeout case."""
        limiter = OpenRouterRateLimiter(tier="free")
        limiter._tokens = 0

        async with limiter.request(timeout=0.05) as ctx:
            assert bool(ctx) is False
            assert ctx._acquired is False

    @pytest.mark.asyncio
    async def test_provider_context_release_on_error(self):
        """Provider context releases token on error."""
        limiter = ProviderRateLimiter("test", rpm=60, burst=5)

        # Freeze refill time
        limiter._last_refill = time.monotonic()

        async with limiter.request(timeout=1.0) as ctx:
            assert ctx._acquired is True
            tokens_after_acquire = limiter._tokens
            # Freeze again
            limiter._last_refill = time.monotonic()

            # Simulate error and release
            ctx.release_on_error()
            limiter._last_refill = time.monotonic()

            assert limiter._tokens > tokens_after_acquire

    @pytest.mark.asyncio
    async def test_nested_context_managers(self):
        """Nested context managers work correctly."""
        limiter1 = ProviderRateLimiter("anthropic")
        limiter2 = ProviderRateLimiter("openai")

        async with limiter1.request(timeout=1.0) as ctx1:
            assert bool(ctx1) is True
            async with limiter2.request(timeout=1.0) as ctx2:
                assert bool(ctx2) is True

    @pytest.mark.asyncio
    async def test_context_release_when_not_acquired(self):
        """Release on error does nothing if not acquired."""
        limiter = ProviderRateLimiter("test", rpm=60, burst=1)
        limiter._tokens = 0

        async with limiter.request(timeout=0.01) as ctx:
            assert ctx._acquired is False
            initial_tokens = limiter._tokens
            ctx.release_on_error()
            # Should not have changed
            assert limiter._tokens == initial_tokens


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestThreadSafety:
    """Tests for thread safety of rate limiters."""

    def test_concurrent_acquisitions_no_race(self):
        """Concurrent acquisitions don't cause race conditions."""
        limiter = ProviderRateLimiter("test", rpm=6000, burst=100)
        results = []
        errors = []

        def acquire_token():
            try:
                # Use synchronous acquire with timeout
                result = limiter._bucket.acquire(timeout=5.0)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=acquire_token) for _ in range(50)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 50
        # Should have at least burst capacity successes
        assert sum(results) >= 50

    def test_concurrent_registry_access(self):
        """Registry is thread-safe for concurrent access."""
        reset_provider_limiters()
        registry = get_provider_registry()
        results = []
        errors = []

        def get_limiter(provider):
            try:
                limiter = registry.get(provider)
                results.append(limiter.provider)
            except Exception as e:
                errors.append(e)

        providers = ["anthropic", "openai", "gemini", "mistral", "grok"]
        threads = []
        for provider in providers * 10:  # 50 total threads
            threads.append(threading.Thread(target=get_limiter, args=(provider,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 50

    def test_backoff_thread_safety(self):
        """ExponentialBackoff is thread-safe."""
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=60.0)
        results = []
        errors = []

        def record_and_reset():
            try:
                for _ in range(10):
                    backoff.record_failure()
                    backoff.reset()
                results.append(True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_and_reset) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10


# ============================================================================
# Environment Variable Override Tests
# ============================================================================


class TestEnvironmentVariables:
    """Tests for environment variable configuration."""

    def test_openrouter_tier_env_override(self):
        """OPENROUTER_TIER env var overrides default tier."""
        with patch.dict(os.environ, {"OPENROUTER_TIER": "premium"}):
            limiter = OpenRouterRateLimiter()
            assert limiter.tier.name == "premium"

    def test_provider_rpm_env_override(self):
        """ARAGORA_{PROVIDER}_RPM env var overrides RPM."""
        with patch.dict(os.environ, {"ARAGORA_TESTPROV_RPM": "999"}):
            limiter = ProviderRateLimiter("testprov")
            assert limiter.requests_per_minute == 999

    def test_provider_burst_env_override(self):
        """ARAGORA_{PROVIDER}_BURST env var overrides burst."""
        with patch.dict(os.environ, {"ARAGORA_TESTPROV_BURST": "77"}):
            limiter = ProviderRateLimiter("testprov")
            assert limiter.burst_size == 77

    def test_env_override_case_insensitive(self):
        """Env var names use uppercase provider name."""
        with patch.dict(os.environ, {"ARAGORA_MYPROVIDER_RPM": "123"}):
            limiter = ProviderRateLimiter("MyProvider")
            assert limiter.requests_per_minute == 123


# ============================================================================
# Statistics Tests
# ============================================================================


class TestStatistics:
    """Tests for rate limiter statistics."""

    @pytest.mark.asyncio
    async def test_stats_track_acquired(self):
        """Stats track acquired count."""
        limiter = ProviderRateLimiter("test", rpm=600, burst=10)

        for _ in range(5):
            await limiter.acquire(timeout=1.0)

        stats = limiter.stats
        assert stats["acquired"] >= 5

    @pytest.mark.asyncio
    async def test_stats_track_rejected(self):
        """Stats track rejected count."""
        limiter = ProviderRateLimiter("test", rpm=60, burst=1)

        # First one succeeds
        await limiter.acquire(timeout=0.1)

        # These should timeout (rejected)
        limiter._tokens = 0
        limiter._last_refill = time.monotonic()  # Freeze refill

        for _ in range(3):
            await limiter.acquire(timeout=0.01)

        stats = limiter.stats
        assert stats["rejected"] >= 2

    def test_stats_include_backoff_info(self):
        """Stats include backoff information."""
        limiter = ProviderRateLimiter("test")

        limiter.record_rate_limit_error(429)
        limiter.record_rate_limit_error(429)

        stats = limiter.stats
        assert stats["backoff_failures"] == 2
        assert stats["is_backing_off"] is True


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_burst_size_handled(self):
        """Zero burst size doesn't cause errors."""
        # Should not crash, though unusual configuration
        limiter = ProviderRateLimiter("test", rpm=60, burst=0)
        assert limiter.burst_size == 0

    def test_very_high_rpm_handled(self):
        """Very high RPM values work correctly."""
        limiter = ProviderRateLimiter("test", rpm=1000000, burst=1000)
        assert limiter.requests_per_minute == 1000000

    def test_negative_timeout_treated_as_zero(self):
        """Negative timeout is treated as zero (non-blocking)."""
        limiter = ProviderRateLimiter("test", rpm=60, burst=5)
        result = limiter._bucket.acquire(timeout=-1.0)
        # Should return immediately (non-blocking)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_acquire_during_heavy_backoff(self):
        """Acquire fails fast when backoff exceeds timeout."""
        limiter = ProviderRateLimiter("test", rpm=60, burst=5)

        # Trigger heavy backoff (many failures)
        for _ in range(10):
            limiter.record_rate_limit_error(429)

        start = time.monotonic()
        result = await limiter.acquire(timeout=0.5)
        elapsed = time.monotonic() - start

        assert result is False
        # Should not have waited the full backoff time
        assert elapsed < 5.0

    def test_record_error_releases_token(self):
        """Recording error releases the token back."""
        limiter = ProviderRateLimiter("test", rpm=60, burst=10)
        limiter._last_refill = time.monotonic()
        initial = limiter._tokens

        # Consume a token
        limiter._bucket._sync_lock.acquire()
        try:
            limiter._bucket._tokens -= 1.0
        finally:
            limiter._bucket._sync_lock.release()

        assert limiter._tokens == initial - 1.0

        # Record error should release it back
        limiter._last_refill = time.monotonic()  # Freeze
        limiter.record_rate_limit_error(429)
        limiter._last_refill = time.monotonic()  # Freeze again

        assert limiter._tokens == initial

    def test_empty_provider_name(self):
        """Empty provider name is handled."""
        limiter = ProviderRateLimiter("")
        assert limiter.provider == ""
        assert limiter.requests_per_minute == 100  # Default

    def test_whitespace_provider_name(self):
        """Whitespace in provider name is preserved."""
        limiter = ProviderRateLimiter("my provider")
        assert limiter.provider == "my provider"


# ============================================================================
# Module-Level Function Tests
# ============================================================================


class TestModuleFunctions:
    """Additional tests for module-level functions."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Reset state before and after tests."""
        reset_provider_limiters()
        yield
        reset_provider_limiters()

    def test_get_provider_limiter_creates_singleton(self):
        """get_provider_limiter returns singleton per provider."""
        limiter1 = get_provider_limiter("anthropic")
        limiter2 = get_provider_limiter("anthropic")

        assert limiter1 is limiter2

    def test_reset_specific_provider_preserves_others(self):
        """Resetting one provider doesn't affect others."""
        anthropic = get_provider_limiter("anthropic")
        openai = get_provider_limiter("openai")

        reset_provider_limiters("anthropic")

        new_anthropic = get_provider_limiter("anthropic")
        same_openai = get_provider_limiter("openai")

        assert new_anthropic is not anthropic  # New instance
        assert same_openai is openai  # Same instance

    def test_reset_all_providers(self):
        """Reset with no args clears all providers."""
        anthropic = get_provider_limiter("anthropic")
        openai = get_provider_limiter("openai")

        reset_provider_limiters()

        new_anthropic = get_provider_limiter("anthropic")
        new_openai = get_provider_limiter("openai")

        assert new_anthropic is not anthropic
        assert new_openai is not openai

    def test_set_openrouter_tier_updates_limiter(self):
        """set_openrouter_tier updates the global limiter."""
        set_openrouter_tier("premium")
        limiter = get_openrouter_limiter()

        assert limiter.tier.name == "premium"


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    @pytest.mark.asyncio
    async def test_full_request_lifecycle(self):
        """Test complete request lifecycle with backoff recovery."""
        limiter = ProviderRateLimiter("test", rpm=600, burst=5)

        # Initial requests succeed
        assert await limiter.acquire(timeout=1.0) is True

        # Simulate rate limit error
        delay = limiter.record_rate_limit_error(429)
        assert delay > 0
        assert limiter.is_backing_off is True

        # Wait less than backoff delay
        await asyncio.sleep(0.1)

        # Record success to reset backoff
        limiter.record_success()
        assert limiter.is_backing_off is False

        # Should be able to acquire again
        assert await limiter.acquire(timeout=1.0) is True

    @pytest.mark.asyncio
    async def test_provider_registry_with_concurrent_providers(self):
        """Test registry with concurrent access to multiple providers."""
        reset_provider_limiters()

        async def make_requests(provider):
            limiter = get_provider_limiter(provider)
            results = []
            for _ in range(3):
                results.append(await limiter.acquire(timeout=1.0))
                await asyncio.sleep(0.01)
            return results

        providers = ["anthropic", "openai", "gemini"]
        all_results = await asyncio.gather(*[make_requests(p) for p in providers])

        # All requests should succeed
        for results in all_results:
            assert all(results)

    @pytest.mark.asyncio
    async def test_header_update_affects_limiting(self):
        """Test that header updates affect rate limiting behavior."""
        limiter = ProviderRateLimiter("test", rpm=600, burst=10)

        # Update headers to indicate low remaining
        limiter.update_from_headers(
            {
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(time.time() + 0.3),
            }
        )

        # Acquire should still work (token bucket has tokens)
        # but stats should reflect API state
        stats = limiter.stats
        assert stats["api_remaining"] == 0


# ============================================================================
# Memory and Resource Tests
# ============================================================================


class TestResourceManagement:
    """Tests for memory and resource management."""

    def test_registry_doesnt_grow_unbounded(self):
        """Registry can be reset to prevent unbounded growth."""
        registry = ProviderRateLimiterRegistry()

        # Add many providers
        for i in range(100):
            registry.get(f"provider_{i}")

        assert len(registry.providers()) == 100

        # Reset clears all
        registry.reset()
        assert len(registry.providers()) == 0

    def test_limiter_stats_dont_grow_unbounded(self):
        """Stats are simple counters, not unbounded lists."""
        limiter = ProviderRateLimiter("test", rpm=60000, burst=1000)

        # Many acquisitions
        for _ in range(1000):
            limiter._bucket.try_acquire()

        stats = limiter.stats
        # Stats should be simple integers, not large data structures
        assert isinstance(stats["acquired"], int)
        assert isinstance(stats["rejected"], int)
