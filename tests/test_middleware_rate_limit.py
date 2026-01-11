"""
Tests for Rate Limiting Middleware.

Tests cover:
- TokenBucket class (token bucket algorithm)
- RateLimitConfig and RateLimitResult dataclasses
- RateLimiter class (multi-key rate limiting)
- RateLimiterRegistry (named limiters)
- TierRateLimiter (subscription tier-based limits)
- Convenience functions and decorators
"""

from __future__ import annotations

import pytest
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from collections import OrderedDict

from aragora.server.middleware.rate_limit import (
    TokenBucket,
    RateLimitConfig,
    RateLimitResult,
    RateLimiter,
    RateLimiterRegistry,
    TierRateLimiter,
    get_rate_limiter,
    cleanup_rate_limiters,
    reset_rate_limiters,
    rate_limit_headers,
    rate_limit,
    get_tier_rate_limiter,
    TIER_RATE_LIMITS,
)
from aragora.services import ServiceRegistry


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_services():
    """Reset ServiceRegistry and rate limiters before each test."""
    reset_rate_limiters()
    yield
    reset_rate_limiters()


@pytest.fixture
def rate_limiter():
    """Create a fresh rate limiter for testing."""
    return RateLimiter(default_limit=60, ip_limit=120)


@pytest.fixture
def tier_limiter():
    """Create a fresh tier rate limiter for testing."""
    return TierRateLimiter()


# ============================================================================
# TokenBucket Tests
# ============================================================================

class TestTokenBucket:
    """Tests for TokenBucket class."""

    def test_initial_tokens_at_burst(self):
        """Test bucket starts with burst_size tokens."""
        bucket = TokenBucket(rate_per_minute=60, burst_size=120)
        assert bucket.tokens == 120.0

    def test_default_burst_size(self):
        """Test default burst size is 2x rate."""
        bucket = TokenBucket(rate_per_minute=60)
        assert bucket.burst_size == 120

    def test_consume_success(self):
        """Test consuming tokens succeeds when available."""
        bucket = TokenBucket(rate_per_minute=60, burst_size=10)
        assert bucket.consume(1) is True
        assert bucket.tokens == 9.0

    def test_consume_multiple_tokens(self):
        """Test consuming multiple tokens at once."""
        bucket = TokenBucket(rate_per_minute=60, burst_size=10)
        assert bucket.consume(5) is True
        assert bucket.tokens == 5.0

    def test_consume_fails_when_empty(self):
        """Test consuming fails when not enough tokens."""
        bucket = TokenBucket(rate_per_minute=60, burst_size=2)
        bucket.consume(2)  # Empty the bucket
        assert bucket.consume(1) is False

    def test_token_refill(self):
        """Test tokens refill over time."""
        bucket = TokenBucket(rate_per_minute=60, burst_size=10)
        bucket.tokens = 0
        bucket.last_refill = time.monotonic() - 1  # 1 second ago

        # After 1 second at 60/min = 1 token refilled
        result = bucket.consume(1)
        assert result is True

    def test_refill_capped_at_burst(self):
        """Test refill doesn't exceed burst size."""
        bucket = TokenBucket(rate_per_minute=60, burst_size=10)
        bucket.last_refill = time.monotonic() - 60  # 1 minute ago

        # Should have refilled 60 tokens but capped at 10
        bucket.consume(1)
        assert bucket.tokens <= 10

    def test_get_retry_after_when_empty(self):
        """Test get_retry_after returns seconds until next token."""
        bucket = TokenBucket(rate_per_minute=60, burst_size=2)
        bucket.tokens = 0

        retry_after = bucket.get_retry_after()
        assert retry_after > 0
        assert retry_after <= 1.0  # At 60/min, should be ~1 second

    def test_get_retry_after_when_has_tokens(self):
        """Test get_retry_after returns 0 when tokens available."""
        bucket = TokenBucket(rate_per_minute=60, burst_size=10)
        assert bucket.get_retry_after() == 0

    def test_remaining_property(self):
        """Test remaining property returns int of tokens."""
        bucket = TokenBucket(rate_per_minute=60, burst_size=10)
        bucket.tokens = 5.7
        assert bucket.remaining == 5

    def test_remaining_property_never_negative(self):
        """Test remaining never returns negative."""
        bucket = TokenBucket(rate_per_minute=60, burst_size=10)
        bucket.tokens = -5
        assert bucket.remaining == 0

    def test_thread_safety(self):
        """Test bucket is thread-safe for concurrent access."""
        bucket = TokenBucket(rate_per_minute=0.001, burst_size=100)  # Very slow refill
        successes = []
        errors = []

        def consume_token():
            try:
                result = bucket.consume(1)
                successes.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=consume_token) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(successes) == 50
        # Should have consumed ~50 tokens from 100 (allow for minor refill)
        assert bucket.tokens <= 51


# ============================================================================
# RateLimitConfig Tests
# ============================================================================

class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RateLimitConfig()
        assert config.requests_per_minute == 60  # DEFAULT_RATE_LIMIT
        assert config.burst_size is None
        assert config.key_type == "ip"
        assert config.enabled is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RateLimitConfig(
            requests_per_minute=30,
            burst_size=60,
            key_type="token",
            enabled=False,
        )
        assert config.requests_per_minute == 30
        assert config.burst_size == 60
        assert config.key_type == "token"
        assert config.enabled is False


# ============================================================================
# RateLimitResult Tests
# ============================================================================

class TestRateLimitResult:
    """Tests for RateLimitResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = RateLimitResult(allowed=True)
        assert result.allowed is True
        assert result.remaining == 0
        assert result.limit == 0
        assert result.retry_after == 0
        assert result.key == ""

    def test_full_result(self):
        """Test fully populated result."""
        result = RateLimitResult(
            allowed=False,
            remaining=0,
            limit=60,
            retry_after=5.0,
            key="ip:192.168.1.1",
        )
        assert result.allowed is False
        assert result.remaining == 0
        assert result.limit == 60
        assert result.retry_after == 5.0
        assert result.key == "ip:192.168.1.1"


# ============================================================================
# RateLimiter Tests
# ============================================================================

class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_init_defaults(self, rate_limiter):
        """Test initialization with defaults."""
        assert rate_limiter.default_limit == 60
        assert rate_limiter.ip_limit == 120

    def test_configure_endpoint(self, rate_limiter):
        """Test configuring endpoint-specific limits."""
        rate_limiter.configure_endpoint("/api/debates", 30, key_type="ip")

        config = rate_limiter.get_config("/api/debates")
        assert config.requests_per_minute == 30
        assert config.key_type == "ip"

    def test_get_config_default(self, rate_limiter):
        """Test get_config returns default for unconfigured endpoints."""
        config = rate_limiter.get_config("/api/unknown")
        assert config.requests_per_minute == rate_limiter.default_limit

    def test_get_config_wildcard(self, rate_limiter):
        """Test get_config matches wildcard patterns."""
        rate_limiter.configure_endpoint("/api/debates/*", 50)

        config = rate_limiter.get_config("/api/debates/123/messages")
        assert config.requests_per_minute == 50

    def test_allow_under_limit(self, rate_limiter):
        """Test allow returns True under limit."""
        result = rate_limiter.allow("192.168.1.1")
        assert result.allowed is True
        assert result.remaining > 0

    def test_allow_over_limit(self, rate_limiter):
        """Test allow returns False when limit exceeded."""
        # Set very low rate with no burst (burst=1 means only 1 token at a time)
        limiter = RateLimiter(default_limit=1, ip_limit=1)
        # Manually configure the bucket to have no burst
        limiter._ip_buckets["192.168.1.1"] = TokenBucket(rate_per_minute=1, burst_size=1)

        # First call consumes the only token
        limiter.allow("192.168.1.1")
        # Second call should be rate limited
        result = limiter.allow("192.168.1.1")

        assert result.allowed is False
        assert result.retry_after > 0

    def test_allow_ip_isolation(self, rate_limiter):
        """Test different IPs have separate limits."""
        limiter = RateLimiter(default_limit=2, ip_limit=2)

        # Exhaust IP 1's limit
        limiter.allow("192.168.1.1")
        limiter.allow("192.168.1.1")
        limiter.allow("192.168.1.1")

        # IP 2 should still have tokens
        result = limiter.allow("192.168.1.2")
        assert result.allowed is True

    def test_allow_with_token_key(self, rate_limiter):
        """Test token-based rate limiting."""
        rate_limiter.configure_endpoint("/api/auth", 10, key_type="token")

        result = rate_limiter.allow(
            "192.168.1.1",
            endpoint="/api/auth",
            token="user-token-123"
        )
        assert result.allowed is True
        assert "token:" in result.key

    def test_allow_with_endpoint_key(self, rate_limiter):
        """Test endpoint-based rate limiting."""
        rate_limiter.configure_endpoint("/api/global", 10, key_type="endpoint")

        result = rate_limiter.allow("192.168.1.1", endpoint="/api/global")
        assert result.allowed is True
        assert "ep:" in result.key

    def test_allow_with_combined_key(self, rate_limiter):
        """Test combined endpoint+IP rate limiting."""
        rate_limiter.configure_endpoint("/api/combined", 10, key_type="combined")

        result = rate_limiter.allow("192.168.1.1", endpoint="/api/combined")
        assert result.allowed is True
        assert "ep:" in result.key and "ip:" in result.key

    def test_allow_disabled_endpoint(self, rate_limiter):
        """Test disabled endpoint always allows."""
        rate_limiter._endpoint_configs["/api/unlimited"] = RateLimitConfig(enabled=False)

        result = rate_limiter.allow("192.168.1.1", endpoint="/api/unlimited")
        assert result.allowed is True
        assert result.limit == 0

    def test_lru_eviction_ip_buckets(self):
        """Test LRU eviction of IP buckets."""
        limiter = RateLimiter(max_entries=9)  # max_entries // 3 = 3 IP buckets

        # Create 4 IP buckets (exceeds limit of 3)
        for i in range(4):
            limiter.allow(f"192.168.1.{i}")

        # First IP should have been evicted
        assert "192.168.1.0" not in limiter._ip_buckets

    def test_cleanup_removes_stale(self, rate_limiter):
        """Test cleanup removes stale entries."""
        # Create some buckets
        rate_limiter.allow("192.168.1.1")
        rate_limiter.allow("192.168.1.2")

        # Make them stale
        for bucket in rate_limiter._ip_buckets.values():
            bucket.last_refill = time.monotonic() - 600  # 10 minutes ago

        removed = rate_limiter.cleanup(max_age_seconds=300)

        assert removed == 2
        assert len(rate_limiter._ip_buckets) == 0

    def test_reset_clears_all(self, rate_limiter):
        """Test reset clears all state."""
        rate_limiter.allow("192.168.1.1")
        rate_limiter.allow("192.168.1.2", endpoint="/api/test")
        rate_limiter.allow("192.168.1.3", endpoint="/api/test", token="tok")

        rate_limiter.reset()

        assert len(rate_limiter._ip_buckets) == 0
        assert len(rate_limiter._token_buckets) == 0
        assert len(rate_limiter._endpoint_buckets) == 0

    def test_get_stats(self, rate_limiter):
        """Test get_stats returns statistics."""
        rate_limiter.allow("192.168.1.1")
        rate_limiter.configure_endpoint("/api/test", 30)

        stats = rate_limiter.get_stats()

        assert stats["ip_buckets"] == 1
        assert stats["default_limit"] == 60
        assert "/api/test" in stats["configured_endpoints"]

    def test_get_client_key_from_handler(self, rate_limiter):
        """Test extracting client key from handler."""
        handler = Mock()
        handler.headers = {"X-Forwarded-For": "10.0.0.1, 192.168.1.1"}

        key = rate_limiter.get_client_key(handler)

        assert key == "10.0.0.1"

    def test_get_client_key_from_client_address(self, rate_limiter):
        """Test extracting client key from client_address."""
        handler = Mock()
        handler.headers = {}
        handler.client_address = ("192.168.1.100", 8080)

        key = rate_limiter.get_client_key(handler)

        assert key == "192.168.1.100"

    def test_get_client_key_anonymous(self, rate_limiter):
        """Test anonymous fallback for client key."""
        key = rate_limiter.get_client_key(None)
        assert key == "anonymous"


# ============================================================================
# RateLimiterRegistry Tests
# ============================================================================

class TestRateLimiterRegistry:
    """Tests for RateLimiterRegistry class."""

    def test_get_default_creates_limiter(self):
        """Test get_default creates default limiter."""
        registry = RateLimiterRegistry()
        limiter = registry.get_default()

        assert isinstance(limiter, RateLimiter)
        assert registry._default_limiter is limiter

    def test_get_default_returns_same(self):
        """Test get_default returns same instance."""
        registry = RateLimiterRegistry()
        limiter1 = registry.get_default()
        limiter2 = registry.get_default()

        assert limiter1 is limiter2

    def test_get_default_has_configured_endpoints(self):
        """Test default limiter has pre-configured endpoints."""
        registry = RateLimiterRegistry()
        limiter = registry.get_default()

        config = limiter.get_config("/api/debates")
        assert config.requests_per_minute == 30

    def test_get_named_limiter(self):
        """Test getting a named limiter."""
        registry = RateLimiterRegistry()
        limiter = registry.get("custom", requests_per_minute=100)

        assert isinstance(limiter, RateLimiter)
        assert limiter.default_limit == 100

    def test_get_named_returns_same(self):
        """Test get returns same named limiter."""
        registry = RateLimiterRegistry()
        limiter1 = registry.get("custom")
        limiter2 = registry.get("custom")

        assert limiter1 is limiter2

    def test_cleanup_all_limiters(self):
        """Test cleanup applies to all limiters."""
        registry = RateLimiterRegistry()
        default = registry.get_default()
        custom = registry.get("custom")

        # Add entries
        default.allow("192.168.1.1")
        custom.allow("192.168.1.2")

        # Make them stale
        for bucket in default._ip_buckets.values():
            bucket.last_refill = time.monotonic() - 600
        for bucket in custom._ip_buckets.values():
            bucket.last_refill = time.monotonic() - 600

        removed = registry.cleanup(max_age_seconds=300)

        assert removed == 2

    def test_reset_all_limiters(self):
        """Test reset clears all limiters."""
        registry = RateLimiterRegistry()
        default = registry.get_default()
        custom = registry.get("custom")

        default.allow("192.168.1.1")
        custom.allow("192.168.1.2")

        registry.reset()

        assert registry._default_limiter is None
        assert len(registry._limiters) == 0


# ============================================================================
# TierRateLimiter Tests
# ============================================================================

class TestTierRateLimiter:
    """Tests for TierRateLimiter class."""

    def test_default_tier_limits(self, tier_limiter):
        """Test default tier limits are set."""
        assert "free" in tier_limiter.tier_limits
        assert "starter" in tier_limiter.tier_limits
        assert "professional" in tier_limiter.tier_limits
        assert "enterprise" in tier_limiter.tier_limits

    def test_get_tier_limits_known(self, tier_limiter):
        """Test getting limits for known tier."""
        rate, burst = tier_limiter.get_tier_limits("professional")
        assert rate == 200
        assert burst == 400

    def test_get_tier_limits_unknown(self, tier_limiter):
        """Test getting limits for unknown tier falls back to free."""
        rate, burst = tier_limiter.get_tier_limits("unknown_tier")
        free_rate, free_burst = tier_limiter.tier_limits["free"]
        assert rate == free_rate
        assert burst == free_burst

    def test_get_tier_limits_case_insensitive(self, tier_limiter):
        """Test tier lookup is case-insensitive."""
        rate1, _ = tier_limiter.get_tier_limits("PROFESSIONAL")
        rate2, _ = tier_limiter.get_tier_limits("professional")
        assert rate1 == rate2

    def test_allow_under_limit(self, tier_limiter):
        """Test allow returns True under tier limit."""
        result = tier_limiter.allow("user-123", "free")
        assert result.allowed is True

    def test_allow_respects_tier_limits(self, tier_limiter):
        """Test different tiers have different limits."""
        # Enterprise should have higher limits than free
        free_result = tier_limiter.allow("free-user", "free")
        enterprise_result = tier_limiter.allow("enterprise-user", "enterprise")

        assert free_result.limit == 10
        assert enterprise_result.limit == 1000

    def test_allow_tier_isolation(self, tier_limiter):
        """Test different tiers have isolated buckets."""
        # Create custom limiter with low limits for testing
        limiter = TierRateLimiter(tier_limits={
            "free": (2, 2),
            "premium": (100, 100),
        })

        # Exhaust free tier limit
        limiter.allow("user-1", "free")
        limiter.allow("user-1", "free")
        free_result = limiter.allow("user-1", "free")

        # Premium should still work
        premium_result = limiter.allow("user-1", "premium")

        assert free_result.allowed is False
        assert premium_result.allowed is True

    def test_allow_key_format(self, tier_limiter):
        """Test result key includes tier."""
        result = tier_limiter.allow("user-123", "starter")
        assert "tier:starter:user-123" == result.key

    def test_get_stats(self, tier_limiter):
        """Test get_stats returns tier statistics."""
        tier_limiter.allow("user-1", "free")
        tier_limiter.allow("user-2", "professional")

        stats = tier_limiter.get_stats()

        assert "tier_buckets" in stats
        assert "tier_limits" in stats
        assert stats["tier_buckets"]["free"] == 1
        assert stats["tier_buckets"]["professional"] == 1

    def test_reset_clears_all_tiers(self, tier_limiter):
        """Test reset clears all tier buckets."""
        tier_limiter.allow("user-1", "free")
        tier_limiter.allow("user-2", "starter")

        tier_limiter.reset()

        stats = tier_limiter.get_stats()
        assert all(count == 0 for count in stats["tier_buckets"].values())


# ============================================================================
# Convenience Functions Tests
# ============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_rate_limiter_default(self):
        """Test get_rate_limiter returns default limiter."""
        limiter = get_rate_limiter()
        assert isinstance(limiter, RateLimiter)

    def test_get_rate_limiter_named(self):
        """Test get_rate_limiter returns named limiter."""
        limiter = get_rate_limiter("custom", requests_per_minute=50)
        assert isinstance(limiter, RateLimiter)
        assert limiter.default_limit == 50

    def test_cleanup_rate_limiters(self):
        """Test cleanup_rate_limiters cleans all."""
        limiter = get_rate_limiter()
        limiter.allow("192.168.1.1")

        # Make stale
        for bucket in limiter._ip_buckets.values():
            bucket.last_refill = time.monotonic() - 600

        removed = cleanup_rate_limiters(max_age_seconds=300)

        assert removed >= 1

    def test_reset_rate_limiters(self):
        """Test reset_rate_limiters resets all."""
        limiter = get_rate_limiter()
        limiter.allow("192.168.1.1")

        reset_rate_limiters()

        # Getting a new default should be fresh
        new_limiter = get_rate_limiter()
        assert len(new_limiter._ip_buckets) == 0


# ============================================================================
# Rate Limit Headers Tests
# ============================================================================

class TestRateLimitHeaders:
    """Tests for rate_limit_headers function."""

    def test_headers_allowed(self):
        """Test headers for allowed request."""
        result = RateLimitResult(
            allowed=True,
            remaining=50,
            limit=60,
            retry_after=0,
        )

        headers = rate_limit_headers(result)

        assert headers["X-RateLimit-Limit"] == "60"
        assert headers["X-RateLimit-Remaining"] == "50"
        assert "Retry-After" not in headers

    def test_headers_rate_limited(self):
        """Test headers for rate limited request."""
        result = RateLimitResult(
            allowed=False,
            remaining=0,
            limit=60,
            retry_after=5.0,
        )

        headers = rate_limit_headers(result)

        assert headers["X-RateLimit-Limit"] == "60"
        assert headers["X-RateLimit-Remaining"] == "0"
        assert "Retry-After" in headers
        assert int(headers["Retry-After"]) >= 5
        assert "X-RateLimit-Reset" in headers


# ============================================================================
# Rate Limit Decorator Tests
# ============================================================================

class TestRateLimitDecorator:
    """Tests for rate_limit decorator."""

    def test_decorator_allows_under_limit(self):
        """Test decorated function executes under limit."""
        @rate_limit(requests_per_minute=100)
        def handler(handler_arg):
            return {"status": "ok"}

        mock_handler = Mock()
        mock_handler.headers = {"X-Forwarded-For": "192.168.1.1"}

        result = handler(mock_handler)

        assert result["status"] == "ok"

    def test_decorator_blocks_over_limit(self):
        """Test decorator returns 429 when rate limited."""
        # Create a fresh limiter with very tight limits
        limiter = get_rate_limiter("test_block_unique", requests_per_minute=1)
        # Set up a bucket with only 1 token
        limiter._ip_buckets["192.168.1.101"] = TokenBucket(rate_per_minute=1, burst_size=1)

        @rate_limit(requests_per_minute=1, burst=1, limiter_name="test_block_unique")
        def handler(handler_arg):
            return {"status": "ok"}

        mock_handler = Mock()
        mock_handler.headers = {"X-Forwarded-For": "192.168.1.101"}

        # First call succeeds (consumes the only token)
        result1 = handler(mock_handler)
        assert result1["status"] == "ok"

        # Second call should be rate limited - returns HandlerResult
        result2 = handler(mock_handler)
        # HandlerResult has status_code attribute
        assert hasattr(result2, "status_code") and result2.status_code == 429

    def test_decorator_per_ip_isolation(self):
        """Test decorator isolates by IP."""
        @rate_limit(requests_per_minute=1, burst=1, limiter_name="test_isolation")
        def handler(handler_arg):
            return {"status": "ok"}

        handler1 = Mock()
        handler1.headers = {"X-Forwarded-For": "192.168.1.1"}

        handler2 = Mock()
        handler2.headers = {"X-Forwarded-For": "192.168.1.2"}

        # Exhaust IP 1
        handler(handler1)
        handler(handler1)

        # IP 2 should still work
        result = handler(handler2)
        assert result["status"] == "ok"


# ============================================================================
# Integration Tests
# ============================================================================

class TestRateLimitIntegration:
    """Integration tests for rate limiting."""

    def test_full_workflow(self):
        """Test complete rate limiting workflow."""
        # Get limiter
        limiter = get_rate_limiter("integration_test", requests_per_minute=5)

        # Make requests
        results = []
        for _ in range(7):
            result = limiter.allow("test-client")
            results.append(result.allowed)

        # First 5 should succeed (burst=10 by default)
        assert results[:5] == [True] * 5

        # Stats should show usage
        stats = limiter.get_stats()
        assert stats["ip_buckets"] == 1

    def test_multiple_limiters_independent(self):
        """Test multiple named limiters are independent."""
        limiter1 = get_rate_limiter("limiter_a", requests_per_minute=2)
        limiter2 = get_rate_limiter("limiter_b", requests_per_minute=2)

        # Exhaust limiter1
        limiter1.allow("client")
        limiter1.allow("client")
        limiter1.allow("client")
        limiter1.allow("client")
        limiter1.allow("client")  # Should be rate limited

        # limiter2 should still work
        result = limiter2.allow("client")
        assert result.allowed is True

    def test_tier_rate_limiting(self):
        """Test tier-based rate limiting."""
        limiter = TierRateLimiter(tier_limits={
            "free": (2, 2),
            "premium": (10, 10),
        })

        # Free user hits limit quickly
        limiter.allow("free-user", "free")
        limiter.allow("free-user", "free")
        free_result = limiter.allow("free-user", "free")

        # Premium user has more capacity
        for _ in range(5):
            limiter.allow("premium-user", "premium")
        premium_result = limiter.allow("premium-user", "premium")

        assert free_result.allowed is False
        assert premium_result.allowed is True
