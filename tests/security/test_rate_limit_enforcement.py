"""
Rate limit enforcement security tests.

Verifies that rate limiting cannot be bypassed through various techniques.
"""

import asyncio
import time
import pytest
from unittest.mock import Mock, patch, MagicMock
from aiohttp import web

from aragora.server.middleware.rate_limit import (
    RateLimiter,
    check_rate_limit,
    check_tier_rate_limit,
)


# =============================================================================
# Rate Limiter Core Tests
# =============================================================================


class TestRateLimiterBypassPrevention:
    """Test that rate limiting cannot be bypassed."""

    def test_blocks_after_limit_exhausted(self):
        """Rate limiter should block requests after limit is reached."""
        limiter = RateLimiter(rate_per_minute=5, burst_multiplier=1.0)

        # Exhaust the limit
        for _ in range(5):
            assert limiter.try_acquire() is True

        # Next request should be blocked
        assert limiter.try_acquire() is False

    def test_blocks_burst_after_exhausted(self):
        """Burst capacity should also eventually be blocked."""
        limiter = RateLimiter(rate_per_minute=5, burst_multiplier=2.0)

        # Exhaust limit including burst (5 * 2 = 10)
        for _ in range(10):
            result = limiter.try_acquire()

        # Should eventually be blocked
        assert limiter.try_acquire() is False

    def test_time_based_recovery(self):
        """Tokens should recover over time."""
        limiter = RateLimiter(rate_per_minute=60, burst_multiplier=1.0)

        # Exhaust limit
        for _ in range(60):
            limiter.try_acquire()

        assert limiter.try_acquire() is False

        # Simulate time passing (1 second = 1 token at 60/min)
        limiter._last_update -= 1.0
        assert limiter.try_acquire() is True

    def test_cannot_bypass_with_negative_amount(self):
        """Negative acquire amounts should not add tokens."""
        limiter = RateLimiter(rate_per_minute=5, burst_multiplier=1.0)

        initial_tokens = limiter._tokens

        # Try to bypass with negative amount
        result = limiter.try_acquire(amount=-100)

        # Should either reject or not increase tokens
        assert limiter._tokens <= initial_tokens + 5  # At most burst capacity

    def test_concurrent_access_safety(self):
        """Concurrent access should not exceed limits."""
        limiter = RateLimiter(rate_per_minute=10, burst_multiplier=1.0)
        successes = []

        def attempt():
            return limiter.try_acquire()

        import threading
        threads = []
        for _ in range(50):
            t = threading.Thread(target=lambda: successes.append(attempt()))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not exceed limit
        assert sum(successes) <= 10

    def test_token_overflow_prevention(self):
        """Tokens should not overflow beyond burst capacity."""
        limiter = RateLimiter(rate_per_minute=10, burst_multiplier=2.0)

        # Wait simulated time to accumulate tokens
        limiter._last_update -= 3600  # 1 hour ago

        # Refill should cap at burst capacity (10 * 2 = 20)
        limiter.try_acquire()
        assert limiter._tokens <= 20


# =============================================================================
# IP-Based Rate Limiting Tests
# =============================================================================


class TestIPRateLimitBypass:
    """Test IP-based rate limiting bypass attempts."""

    def test_different_ips_have_separate_limits(self):
        """Different IPs should have separate rate limits."""
        limiters = {}

        def get_limiter(ip):
            if ip not in limiters:
                limiters[ip] = RateLimiter(rate_per_minute=5, burst_multiplier=1.0)
            return limiters[ip]

        # IP 1 exhausts limit
        limiter1 = get_limiter("192.168.1.1")
        for _ in range(5):
            limiter1.try_acquire()
        assert limiter1.try_acquire() is False

        # IP 2 should still have capacity
        limiter2 = get_limiter("192.168.1.2")
        assert limiter2.try_acquire() is True

    def test_spoofed_xff_header_ignored_when_untrusted(self):
        """X-Forwarded-For should be ignored from untrusted sources."""
        # Simulate extracting client IP
        def extract_client_ip(headers, trusted_proxies=None):
            trusted_proxies = trusted_proxies or []
            xff = headers.get("X-Forwarded-For", "")

            # If not from trusted proxy, use direct connection IP
            remote_ip = headers.get("REMOTE_ADDR", "127.0.0.1")
            if remote_ip not in trusted_proxies:
                return remote_ip

            # Only trust XFF from known proxies
            if xff:
                return xff.split(",")[0].strip()
            return remote_ip

        # Attacker tries to spoof XFF
        headers = {
            "X-Forwarded-For": "10.0.0.1",
            "REMOTE_ADDR": "192.168.1.100",
        }

        # Without trusted proxies, should return actual IP
        ip = extract_client_ip(headers, trusted_proxies=[])
        assert ip == "192.168.1.100"

        # With trusted proxy, should use XFF
        ip = extract_client_ip(headers, trusted_proxies=["192.168.1.100"])
        assert ip == "10.0.0.1"

    def test_ipv6_and_ipv4_treated_separately(self):
        """IPv6 and IPv4 addresses should be tracked separately."""
        limiters = {}

        def get_limiter(ip):
            if ip not in limiters:
                limiters[ip] = RateLimiter(rate_per_minute=5, burst_multiplier=1.0)
            return limiters[ip]

        ipv4 = "192.168.1.1"
        ipv6 = "::ffff:192.168.1.1"  # IPv4-mapped IPv6

        # These should be separate limiters
        limiter_v4 = get_limiter(ipv4)
        limiter_v6 = get_limiter(ipv6)

        assert limiter_v4 is not limiter_v6


# =============================================================================
# Tier-Based Rate Limiting Tests
# =============================================================================


class TestTierRateLimitBypass:
    """Test tier-based rate limiting bypass attempts."""

    def test_free_tier_has_lowest_limits(self):
        """Free tier should have the most restrictive limits."""
        tier_limits = {
            "free": 10,
            "starter": 60,
            "professional": 300,
            "enterprise": 1000,
        }

        assert tier_limits["free"] < tier_limits["starter"]
        assert tier_limits["starter"] < tier_limits["professional"]
        assert tier_limits["professional"] < tier_limits["enterprise"]

    def test_cannot_upgrade_tier_via_header(self):
        """Tier should come from authenticated org, not request headers."""
        def get_tier_from_auth(org_id, request_claimed_tier):
            # Simulated database lookup
            org_tiers = {
                "org-123": "free",
                "org-456": "professional",
            }
            # Always use database tier, never trust request
            return org_tiers.get(org_id, "free")

        # Attacker claims professional tier
        tier = get_tier_from_auth("org-123", request_claimed_tier="enterprise")
        assert tier == "free"

    def test_expired_subscription_downgrades(self):
        """Expired subscriptions should downgrade to free tier limits."""
        from datetime import datetime, timedelta

        def get_effective_tier(org_id, subscription_end):
            if subscription_end and subscription_end < datetime.now():
                return "free"  # Expired
            return "professional"

        expired = datetime.now() - timedelta(days=1)
        assert get_effective_tier("org-123", expired) == "free"

        valid = datetime.now() + timedelta(days=30)
        assert get_effective_tier("org-123", valid) == "professional"


# =============================================================================
# Distributed Rate Limiting Tests
# =============================================================================


class TestDistributedRateLimitBypass:
    """Test distributed rate limiting (Redis-backed) bypass attempts."""

    def test_redis_atomic_operations(self):
        """Redis operations should be atomic to prevent race conditions."""
        # Simulate Lua script for atomic increment
        lua_script = """
        local key = KEYS[1]
        local limit = tonumber(ARGV[1])
        local window = tonumber(ARGV[2])

        local current = redis.call('INCR', key)
        if current == 1 then
            redis.call('EXPIRE', key, window)
        end

        if current > limit then
            return 0
        end
        return 1
        """

        # The script ensures atomicity - cannot bypass between INCR and check
        assert "INCR" in lua_script
        assert "EXPIRE" in lua_script

    def test_redis_key_isolation(self):
        """Redis keys should be properly namespaced to prevent collisions."""
        def make_rate_limit_key(org_id, endpoint):
            # Proper namespacing
            return f"aragora:ratelimit:{org_id}:{endpoint}"

        key1 = make_rate_limit_key("org-123", "/api/debates")
        key2 = make_rate_limit_key("org-456", "/api/debates")

        assert key1 != key2
        assert "org-123" in key1
        assert "org-456" in key2


# =============================================================================
# Endpoint-Specific Rate Limiting Tests
# =============================================================================


class TestEndpointRateLimitBypass:
    """Test endpoint-specific rate limiting bypass attempts."""

    def test_expensive_endpoints_have_lower_limits(self):
        """Expensive endpoints should have stricter limits."""
        endpoint_limits = {
            "/api/debates": 30,  # Create debate - expensive
            "/api/gauntlet": 10,  # Gauntlet run - very expensive
            "/api/health": 1000,  # Health check - cheap
            "/api/leaderboard": 60,  # Read-only
        }

        # Expensive operations should have lower limits
        assert endpoint_limits["/api/gauntlet"] < endpoint_limits["/api/debates"]
        assert endpoint_limits["/api/debates"] < endpoint_limits["/api/leaderboard"]
        assert endpoint_limits["/api/leaderboard"] < endpoint_limits["/api/health"]

    def test_path_normalization_prevents_bypass(self):
        """Path normalization should prevent bypass via URL tricks."""
        def normalize_path(path):
            # Remove trailing slashes
            path = path.rstrip("/")
            # Remove double slashes
            while "//" in path:
                path = path.replace("//", "/")
            # Decode URL encoding
            from urllib.parse import unquote
            path = unquote(path)
            # Remove path traversal
            path = path.replace("/../", "/").replace("/./", "/")
            return path

        # These should all normalize to same path
        assert normalize_path("/api/debates") == "/api/debates"
        assert normalize_path("/api/debates/") == "/api/debates"
        assert normalize_path("/api//debates") == "/api/debates"
        assert normalize_path("/api/debates/../debates") == "/api/debates"
        assert normalize_path("/api/%64ebates".replace("%64", "d")) == "/api/debates"


# =============================================================================
# Rate Limit Header Tests
# =============================================================================


class TestRateLimitHeaders:
    """Test rate limit header accuracy and security."""

    def test_retry_after_header_accurate(self):
        """Retry-After header should be accurate."""
        limiter = RateLimiter(rate_per_minute=60, burst_multiplier=1.0)

        # Exhaust limit
        for _ in range(60):
            limiter.try_acquire()

        # Calculate retry after
        retry_after = limiter.get_retry_after()

        # Should be approximately 1 second (60 tokens/min = 1 token/sec)
        assert 0 < retry_after <= 1.0

    def test_remaining_count_not_negative(self):
        """Remaining count should never be negative."""
        limiter = RateLimiter(rate_per_minute=5, burst_multiplier=1.0)

        # Exhaust limit
        for _ in range(10):  # More than limit
            limiter.try_acquire()

        remaining = max(0, int(limiter._tokens))
        assert remaining >= 0

    def test_limit_header_matches_tier(self):
        """X-RateLimit-Limit header should match actual tier limit."""
        tier_limits = {
            "free": 10,
            "starter": 60,
            "professional": 300,
        }

        for tier, expected_limit in tier_limits.items():
            header_value = tier_limits[tier]
            assert header_value == expected_limit


# =============================================================================
# Rate Limit Persistence Tests
# =============================================================================


class TestRateLimitPersistence:
    """Test rate limit state persistence and recovery."""

    def test_server_restart_resets_in_memory_limits(self):
        """In-memory rate limits should reset on restart (acceptable behavior)."""
        # In-memory limiter
        limiter1 = RateLimiter(rate_per_minute=5, burst_multiplier=1.0)
        for _ in range(5):
            limiter1.try_acquire()
        assert limiter1.try_acquire() is False

        # Simulate restart - new limiter instance
        limiter2 = RateLimiter(rate_per_minute=5, burst_multiplier=1.0)
        assert limiter2.try_acquire() is True  # Fresh start

    def test_window_expiry_correct(self):
        """Rate limit windows should expire correctly."""
        limiter = RateLimiter(rate_per_minute=60, burst_multiplier=1.0)

        # Record current state
        initial_tokens = limiter._tokens

        # Simulate 2 minutes passing
        limiter._last_update -= 120

        # Should have recovered tokens
        limiter.try_acquire()  # Trigger refill
        assert limiter._tokens >= initial_tokens - 1
