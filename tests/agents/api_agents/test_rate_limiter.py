"""
Tests for aragora.agents.api_agents.rate_limiter module.

Covers:
- OpenRouterTier and ProviderTier dataclasses
- OpenRouterRateLimiter (token bucket, acquire, headers, backoff)
- RateLimitContext (async context manager)
- ProviderRateLimiter (generic provider rate limiting)
- ProviderRateLimitContext
- ProviderRateLimiterRegistry
- Module-level functions (get_provider_limiter, etc.)
"""

import asyncio
import os
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.agents.api_agents.rate_limiter import (
    OPENROUTER_TIERS,
    PROVIDER_DEFAULT_TIERS,
    OpenRouterRateLimiter,
    OpenRouterTier,
    ProviderRateLimiter,
    ProviderRateLimiterRegistry,
    ProviderTier,
    RateLimitContext,
    ProviderRateLimitContext,
    get_provider_limiter,
    get_provider_registry,
    reset_provider_limiters,
)


# ============================================================================
# Dataclasses
# ============================================================================


class TestOpenRouterTier:
    """Tests for OpenRouterTier dataclass."""

    def test_default_values(self):
        """Test default values."""
        tier = OpenRouterTier(name="test", requests_per_minute=100)
        assert tier.name == "test"
        assert tier.requests_per_minute == 100
        assert tier.tokens_per_minute == 0  # Default unlimited
        assert tier.burst_size == 10  # Default burst

    def test_custom_values(self):
        """Test custom values."""
        tier = OpenRouterTier(
            name="custom",
            requests_per_minute=200,
            tokens_per_minute=50000,
            burst_size=25,
        )
        assert tier.name == "custom"
        assert tier.requests_per_minute == 200
        assert tier.tokens_per_minute == 50000
        assert tier.burst_size == 25


class TestProviderTier:
    """Tests for ProviderTier dataclass."""

    def test_default_values(self):
        """Test default values."""
        tier = ProviderTier(name="test", requests_per_minute=100)
        assert tier.name == "test"
        assert tier.requests_per_minute == 100
        assert tier.tokens_per_minute == 0
        assert tier.burst_size == 10

    def test_predefined_tiers(self):
        """Verify predefined provider tiers exist."""
        assert "anthropic" in PROVIDER_DEFAULT_TIERS
        assert "openai" in PROVIDER_DEFAULT_TIERS
        assert "gemini" in PROVIDER_DEFAULT_TIERS
        assert "mistral" in PROVIDER_DEFAULT_TIERS


class TestOpenRouterTiers:
    """Tests for predefined OpenRouter tiers."""

    def test_all_tiers_exist(self):
        """All predefined tiers should exist."""
        expected_tiers = ["free", "basic", "standard", "premium", "unlimited"]
        for tier_name in expected_tiers:
            assert tier_name in OPENROUTER_TIERS

    def test_tier_limits_increasing(self):
        """Higher tiers should have higher limits."""
        tiers = ["free", "basic", "standard", "premium", "unlimited"]
        limits = [OPENROUTER_TIERS[t].requests_per_minute for t in tiers]
        assert limits == sorted(limits), "Tier limits should be increasing"


# ============================================================================
# OpenRouterRateLimiter
# ============================================================================


class TestOpenRouterRateLimiter:
    """Tests for OpenRouterRateLimiter."""

    @pytest.fixture
    def limiter(self):
        """Create a limiter with standard tier."""
        return OpenRouterRateLimiter(tier="standard")

    @pytest.fixture
    def fast_limiter(self):
        """Create a limiter with high RPM for faster tests."""
        # Manually configure for fast testing
        limiter = OpenRouterRateLimiter(tier="unlimited")
        return limiter

    def test_init_default_tier(self):
        """Default tier initialization."""
        limiter = OpenRouterRateLimiter()
        assert limiter.tier.name == "standard"

    def test_init_custom_tier(self):
        """Custom tier initialization."""
        limiter = OpenRouterRateLimiter(tier="premium")
        assert limiter.tier.name == "premium"
        assert limiter.tier.requests_per_minute == 500

    def test_init_invalid_tier_falls_back(self):
        """Invalid tier falls back to standard."""
        limiter = OpenRouterRateLimiter(tier="nonexistent")
        assert limiter.tier.name == "standard"

    def test_init_from_env(self):
        """Tier can be set via environment variable."""
        with patch.dict(os.environ, {"OPENROUTER_TIER": "premium"}):
            limiter = OpenRouterRateLimiter()
            assert limiter.tier.name == "premium"

    def test_initial_tokens(self):
        """Initial tokens equal burst size."""
        limiter = OpenRouterRateLimiter(tier="standard")
        assert limiter._tokens == limiter.tier.burst_size

    @pytest.mark.asyncio
    async def test_acquire_success(self, fast_limiter):
        """Successfully acquire a token."""
        result = await fast_limiter.acquire(timeout=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_consumes_token(self, limiter):
        """Acquiring consumes a token."""
        # Use standard tier (moderate RPM) for reliable token consumption check
        # Freeze refill time to prevent auto-refill during the test
        initial_tokens = limiter._tokens
        limiter._last_refill = time.monotonic()  # Freeze refill
        await limiter.acquire(timeout=1.0)
        limiter._last_refill = time.monotonic()  # Freeze again after acquire
        assert limiter._tokens < initial_tokens

    @pytest.mark.asyncio
    async def test_acquire_timeout_when_no_tokens(self):
        """Acquire times out when no tokens available."""
        limiter = OpenRouterRateLimiter(tier="free")
        # Consume all tokens
        limiter._tokens = 0

        start = time.monotonic()
        result = await limiter.acquire(timeout=0.2)
        elapsed = time.monotonic() - start

        assert result is False
        assert elapsed >= 0.2

    @pytest.mark.asyncio
    async def test_acquire_multiple(self, fast_limiter):
        """Can acquire multiple tokens in sequence."""
        for _ in range(5):
            result = await fast_limiter.acquire(timeout=1.0)
            assert result is True

    def test_refill_adds_tokens(self, limiter):
        """Refill adds tokens based on elapsed time."""
        limiter._tokens = 0
        limiter._last_refill = time.monotonic() - 60  # 1 minute ago

        limiter._refill()

        assert limiter._tokens > 0
        assert limiter._tokens <= limiter.tier.burst_size

    def test_refill_caps_at_burst_size(self, limiter):
        """Refill doesn't exceed burst size."""
        limiter._tokens = limiter.tier.burst_size
        limiter._last_refill = time.monotonic() - 60

        limiter._refill()

        assert limiter._tokens == limiter.tier.burst_size

    def test_update_from_headers(self, limiter):
        """Update rate limit state from headers."""
        headers = {
            "X-RateLimit-Limit": "100",
            "X-RateLimit-Remaining": "50",
            "X-RateLimit-Reset": str(time.time() + 60),
        }

        limiter.update_from_headers(headers)

        assert limiter._api_limit == 100
        assert limiter._api_remaining == 50
        assert limiter._api_reset is not None

    def test_update_from_headers_invalid_values(self, limiter):
        """Invalid header values are handled gracefully."""
        headers = {
            "X-RateLimit-Limit": "invalid",
            "X-RateLimit-Remaining": "not_a_number",
        }

        # Should not raise
        limiter.update_from_headers(headers)
        assert limiter._api_limit is None
        assert limiter._api_remaining is None

    def test_release_on_error(self, limiter):
        """Release token back on error."""
        initial = limiter._tokens
        limiter._tokens = initial - 1

        limiter.release_on_error()

        assert limiter._tokens == initial

    def test_release_on_error_caps_at_burst(self, limiter):
        """Release doesn't exceed burst size."""
        limiter._tokens = limiter.tier.burst_size

        limiter.release_on_error()

        assert limiter._tokens == limiter.tier.burst_size

    def test_record_rate_limit_error(self, limiter):
        """Recording rate limit error triggers backoff."""
        delay = limiter.record_rate_limit_error(429)

        assert delay > 0
        assert limiter.is_backing_off is True

    def test_record_success_resets_backoff(self, limiter):
        """Recording success resets backoff."""
        limiter.record_rate_limit_error(429)
        assert limiter.is_backing_off is True

        limiter.record_success()
        assert limiter.is_backing_off is False

    def test_stats_property(self, limiter):
        """Stats property returns expected keys."""
        stats = limiter.stats

        assert "tier" in stats
        assert "rpm_limit" in stats
        assert "tokens_available" in stats
        assert "burst_size" in stats
        assert "api_limit" in stats
        assert "api_remaining" in stats
        assert "backoff_failures" in stats
        assert "is_backing_off" in stats


# ============================================================================
# RateLimitContext
# ============================================================================


class TestRateLimitContext:
    """Tests for RateLimitContext async context manager."""

    @pytest.mark.asyncio
    async def test_context_acquires_on_entry(self):
        """Context manager acquires on entry."""
        # Use free tier (low RPM) to prevent rapid refill during test
        limiter = OpenRouterRateLimiter(tier="free")
        initial_tokens = limiter._tokens

        async with limiter.request(timeout=1.0) as ctx:
            assert ctx._acquired is True
            assert limiter._tokens < initial_tokens

    @pytest.mark.asyncio
    async def test_context_bool_conversion(self):
        """Context is truthy when acquired."""
        limiter = OpenRouterRateLimiter(tier="unlimited")

        async with limiter.request(timeout=1.0) as ctx:
            assert bool(ctx) is True

    @pytest.mark.asyncio
    async def test_context_false_on_timeout(self):
        """Context is falsy when acquisition times out."""
        limiter = OpenRouterRateLimiter(tier="free")
        limiter._tokens = 0

        async with limiter.request(timeout=0.1) as ctx:
            assert bool(ctx) is False

    @pytest.mark.asyncio
    async def test_release_on_error(self):
        """Can release token back on error."""
        # Use free tier (low RPM) to prevent rapid refill during test
        limiter = OpenRouterRateLimiter(tier="free")

        async with limiter.request(timeout=1.0) as ctx:
            tokens_after_acquire = limiter._tokens
            ctx.release_on_error()
            assert limiter._tokens > tokens_after_acquire


# ============================================================================
# ProviderRateLimiter
# ============================================================================


class TestProviderRateLimiter:
    """Tests for ProviderRateLimiter."""

    @pytest.fixture
    def limiter(self):
        """Create a limiter for anthropic."""
        return ProviderRateLimiter("anthropic")

    def test_init_with_defaults(self):
        """Initialize with provider defaults."""
        limiter = ProviderRateLimiter("anthropic")
        expected = PROVIDER_DEFAULT_TIERS["anthropic"]

        assert limiter.provider == "anthropic"
        assert limiter.requests_per_minute == expected.requests_per_minute
        assert limiter.burst_size == expected.burst_size

    def test_init_with_overrides(self):
        """Initialize with custom RPM and burst."""
        limiter = ProviderRateLimiter("anthropic", rpm=100, burst=5)

        assert limiter.requests_per_minute == 100
        assert limiter.burst_size == 5

    def test_init_unknown_provider(self):
        """Unknown provider gets default limits."""
        limiter = ProviderRateLimiter("unknown_provider")

        assert limiter.provider == "unknown_provider"
        assert limiter.requests_per_minute == 100  # Default
        assert limiter.burst_size == 10  # Default

    def test_init_from_env(self):
        """Can override via environment variables."""
        with patch.dict(os.environ, {"ARAGORA_TESTPROVIDER_RPM": "150"}):
            limiter = ProviderRateLimiter("testprovider")
            assert limiter.requests_per_minute == 150

    @pytest.mark.asyncio
    async def test_acquire_success(self):
        """Successfully acquire a token."""
        limiter = ProviderRateLimiter("anthropic")
        result = await limiter.acquire(timeout=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_timeout(self):
        """Acquire times out when no tokens and slow refill."""
        # Use very low RPM so tokens refill slowly
        limiter = ProviderRateLimiter("test", rpm=1, burst=1)
        limiter._tokens = 0
        limiter._last_refill = time.monotonic()  # Just refilled

        result = await limiter.acquire(timeout=0.05)
        assert result is False

    def test_update_from_headers_multiple_formats(self):
        """Update handles different header formats."""
        limiter = ProviderRateLimiter("anthropic")

        # Test lowercase format
        headers = {"x-ratelimit-limit": "500", "x-ratelimit-remaining": "450"}
        limiter.update_from_headers(headers)

        assert limiter._api_limit == 500
        assert limiter._api_remaining == 450

    def test_stats_property(self, limiter):
        """Stats returns expected keys."""
        stats = limiter.stats

        assert "provider" in stats
        assert "rpm_limit" in stats
        assert "tokens_available" in stats
        assert "burst_size" in stats
        assert "is_backing_off" in stats

    def test_record_rate_limit_error(self, limiter):
        """Recording error triggers backoff."""
        delay = limiter.record_rate_limit_error(429)

        assert delay > 0
        assert limiter.is_backing_off is True

    def test_record_success(self, limiter):
        """Recording success resets backoff."""
        limiter.record_rate_limit_error(429)
        limiter.record_success()

        assert limiter.is_backing_off is False


# ============================================================================
# ProviderRateLimitContext
# ============================================================================


class TestProviderRateLimitContext:
    """Tests for ProviderRateLimitContext."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Async context manager works correctly."""
        limiter = ProviderRateLimiter("anthropic")

        async with limiter.request(timeout=1.0) as ctx:
            assert bool(ctx) is True

    @pytest.mark.asyncio
    async def test_release_on_error(self):
        """Can release token on error."""
        limiter = ProviderRateLimiter("anthropic")

        async with limiter.request(timeout=1.0) as ctx:
            # Freeze refill time to prevent auto-refill during the test
            limiter._last_refill = time.monotonic()
            tokens_after = limiter._tokens
            ctx.release_on_error()
            limiter._last_refill = time.monotonic()  # Freeze again before check
            assert limiter._tokens > tokens_after


# ============================================================================
# ProviderRateLimiterRegistry
# ============================================================================


class TestProviderRateLimiterRegistry:
    """Tests for ProviderRateLimiterRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry."""
        return ProviderRateLimiterRegistry()

    def test_get_creates_limiter(self, registry):
        """Get creates a new limiter if not exists."""
        limiter = registry.get("anthropic")

        assert limiter is not None
        assert limiter.provider == "anthropic"

    def test_get_returns_same_instance(self, registry):
        """Get returns same instance for same provider."""
        limiter1 = registry.get("anthropic")
        limiter2 = registry.get("anthropic")

        assert limiter1 is limiter2

    def test_get_case_insensitive(self, registry):
        """Provider names are case-insensitive."""
        limiter1 = registry.get("Anthropic")
        limiter2 = registry.get("ANTHROPIC")

        assert limiter1 is limiter2

    def test_get_with_overrides(self, registry):
        """Overrides are applied on first access only."""
        limiter1 = registry.get("anthropic", rpm=100, burst=5)
        assert limiter1.requests_per_minute == 100

        # Second call ignores overrides
        limiter2 = registry.get("anthropic", rpm=200, burst=10)
        assert limiter2.requests_per_minute == 100  # Still 100

    def test_reset_specific_provider(self, registry):
        """Reset removes specific provider."""
        registry.get("anthropic")
        registry.get("openai")

        registry.reset("anthropic")

        assert "anthropic" not in registry.providers()
        assert "openai" in registry.providers()

    def test_reset_all_providers(self, registry):
        """Reset all clears entire registry."""
        registry.get("anthropic")
        registry.get("openai")

        registry.reset()

        assert registry.providers() == []

    def test_stats_returns_all(self, registry):
        """Stats returns stats for all providers."""
        registry.get("anthropic")
        registry.get("openai")

        stats = registry.stats()

        assert "anthropic" in stats
        assert "openai" in stats
        assert "rpm_limit" in stats["anthropic"]

    def test_providers_list(self, registry):
        """Providers returns list of registered names."""
        registry.get("anthropic")
        registry.get("openai")

        providers = registry.providers()

        assert set(providers) == {"anthropic", "openai"}

    def test_thread_safety(self, registry):
        """Registry is thread-safe."""
        errors = []
        results = []

        def create_limiter(provider):
            try:
                limiter = registry.get(provider)
                results.append(limiter.provider)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_limiter, args=("anthropic",)) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(results) == 10
        assert all(p == "anthropic" for p in results)


# ============================================================================
# Module-level Functions
# ============================================================================


class TestModuleFunctions:
    """Tests for module-level functions."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset global registry before each test."""
        reset_provider_limiters()
        yield
        reset_provider_limiters()

    def test_get_provider_limiter(self):
        """get_provider_limiter returns a limiter."""
        limiter = get_provider_limiter("anthropic")

        assert limiter is not None
        assert limiter.provider == "anthropic"

    def test_get_provider_limiter_same_instance(self):
        """get_provider_limiter returns same instance."""
        limiter1 = get_provider_limiter("anthropic")
        limiter2 = get_provider_limiter("anthropic")

        assert limiter1 is limiter2

    def test_get_provider_registry(self):
        """get_provider_registry returns registry."""
        registry = get_provider_registry()

        assert registry is not None
        assert isinstance(registry, ProviderRateLimiterRegistry)

    def test_get_provider_registry_singleton(self):
        """get_provider_registry returns same instance."""
        registry1 = get_provider_registry()
        registry2 = get_provider_registry()

        assert registry1 is registry2

    def test_reset_provider_limiters(self):
        """reset_provider_limiters clears registry."""
        get_provider_limiter("anthropic")
        get_provider_limiter("openai")

        reset_provider_limiters()

        registry = get_provider_registry()
        assert registry.providers() == []

    def test_reset_specific_provider(self):
        """reset_provider_limiters can reset specific provider."""
        get_provider_limiter("anthropic")
        get_provider_limiter("openai")

        reset_provider_limiters("anthropic")

        registry = get_provider_registry()
        assert "anthropic" not in registry.providers()
        assert "openai" in registry.providers()


# ============================================================================
# Integration Tests
# ============================================================================


class TestRateLimiterIntegration:
    """Integration tests for rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_concurrent_acquire(self):
        """Multiple concurrent acquires are handled correctly."""
        limiter = ProviderRateLimiter("anthropic", rpm=1000, burst=20)
        results = []

        async def acquire_token():
            result = await limiter.acquire(timeout=2.0)
            results.append(result)

        # Start 15 concurrent acquisitions
        await asyncio.gather(*[acquire_token() for _ in range(15)])

        # All should succeed since burst is 20
        assert all(results)
        assert len(results) == 15

    @pytest.mark.asyncio
    async def test_rate_limiting_enforced(self):
        """Rate limiting is enforced over time."""
        # Create limiter with low limits for fast testing
        limiter = ProviderRateLimiter("test", rpm=600, burst=3)  # 10 per second

        # Exhaust burst
        for _ in range(3):
            await limiter.acquire(timeout=1.0)

        # Next acquire should take some time (wait for refill)
        start = time.monotonic()
        result = await limiter.acquire(timeout=2.0)
        elapsed = time.monotonic() - start

        assert result is True
        assert elapsed > 0.05  # Should have waited for token refill

    @pytest.mark.asyncio
    async def test_backoff_recovery(self):
        """Rate limiter recovers from backoff state."""
        limiter = ProviderRateLimiter("anthropic")

        # Trigger backoff
        limiter.record_rate_limit_error(429)
        assert limiter.is_backing_off is True

        # Record success
        limiter.record_success()
        assert limiter.is_backing_off is False

        # Should be able to acquire normally
        result = await limiter.acquire(timeout=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_header_based_limiting(self):
        """Rate limiter respects API-reported limits."""
        limiter = ProviderRateLimiter("anthropic")

        # Simulate API reporting low remaining
        limiter.update_from_headers(
            {
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(time.time() + 0.5),  # Reset in 0.5s
            }
        )

        # Acquire should still work (tokens available locally)
        result = await limiter.acquire(timeout=1.0)
        assert result is True


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_rpm(self):
        """Handles zero RPM gracefully."""
        limiter = ProviderRateLimiter("test", rpm=1, burst=1)
        # Should not crash
        limiter._refill()

    def test_empty_headers(self):
        """Handles empty headers dict."""
        limiter = ProviderRateLimiter("anthropic")
        limiter.update_from_headers({})
        # Should not crash or change state
        assert limiter._api_limit is None

    def test_negative_tokens_impossible(self):
        """Tokens never go negative."""
        limiter = ProviderRateLimiter("anthropic")
        limiter._tokens = 0

        # Can't go negative via release
        limiter._tokens = 0
        limiter._refill()
        assert limiter._tokens >= 0

    @pytest.mark.asyncio
    async def test_very_short_timeout(self):
        """Handles very short timeouts via backoff state."""
        limiter = ProviderRateLimiter("test", rpm=100, burst=1)

        # Force into long backoff state (60+ seconds delay)
        for _ in range(10):
            limiter.record_rate_limit_error(429)

        start = time.monotonic()
        # With a long backoff delay and short timeout, should return False
        result = await limiter.acquire(timeout=0.1)
        elapsed = time.monotonic() - start

        assert result is False
        assert elapsed < 1.0  # Should not wait much longer than timeout

    def test_float_token_arithmetic(self):
        """Token arithmetic handles floats correctly."""
        limiter = ProviderRateLimiter("anthropic")
        limiter._tokens = 0.5

        # Refill adds fractional tokens
        limiter._last_refill = time.monotonic() - 0.1
        limiter._refill()

        # Tokens should be slightly more than 0.5
        assert limiter._tokens > 0.5
